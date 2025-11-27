import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
import cv2
import scipy.linalg


class KalmanFilter:
    def __init__(self):
        self._motion_mat = np.eye(8, 8)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1
        self._update_mat = np.eye(4, 8)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros_like(measurement)]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             np.dot(covariance, self._update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = 'tentative' if n_init > 0 else 'confirmed'
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self._n_init = n_init
        self._max_age = max_age

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == 'tentative' and self.hits >= self._n_init:
            self.state = 'confirmed'

    def mark_missed(self):
        if self.state == 'tentative':
            self.state = 'deleted'
        elif self.time_since_update > self._max_age:
            self.state = 'deleted'

    def is_tentative(self):
        return self.state == 'tentative'

    def is_confirmed(self):
        return self.state == 'confirmed'

    def is_deleted(self):
        return self.state == 'deleted'


class Detection:
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class NearestNeighborDistanceMetric:
    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "cosine":
            self._metric = self._cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def _cosine_distance(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Нормализация
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)

        x_norm[x_norm == 0] = 1e-10
        y_norm[y_norm == 0] = 1e-10

        x = x / x_norm
        y = y / y_norm

        return 1. - np.dot(x, y.T)

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(features), len(targets)), dtype=np.float32)
        for i, feature in enumerate(features):
            for j, target in enumerate(targets):
                if target in self.samples and len(self.samples[target]) > 0:
                    cost_matrix[i, j] = self._metric([feature], self.samples[target])[0]
                else:
                    cost_matrix[i, j] = 1.0  # Максимальная дистанция
        return cost_matrix


class Tracker:
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management."""
        try:
            # Валидация входных данных
            if detections is None:
                detections = []

            # Run matching cascade.
            matches, unmatched_tracks, unmatched_detections = self._match(detections)

            # Update track set.
            for track_idx, detection_idx in matches:
                # Проверяем валидность индексов
                if (0 <= track_idx < len(self.tracks) and
                        0 <= detection_idx < len(detections)):
                    self.tracks[track_idx].update(self.kf, detections[detection_idx])
                else:
                    print(f"Invalid match indices: track_idx={track_idx}, detection_idx={detection_idx}")

            for track_idx in unmatched_tracks:
                if 0 <= track_idx < len(self.tracks):
                    self.tracks[track_idx].mark_missed()
                else:
                    print(f"Invalid unmatched track index: {track_idx}")

            for detection_idx in unmatched_detections:
                if 0 <= detection_idx < len(detections):
                    self._initiate_track(detections[detection_idx])
                else:
                    print(f"Invalid unmatched detection index: {detection_idx}")

            # Remove deleted tracks.
            self.tracks = [t for t in self.tracks if not t.is_deleted()]

        except Exception as e:
            print(f"Error in tracker update: {e}")
            import traceback
            traceback.print_exc()
            # В случае ошибки просто удаляем невалидные треки
            self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _match(self, detections):
        if len(detections) == 0:
            return [], [], []

        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))

        # Разделяем треки на подтвержденные и неподтвержденные
        confirmed_tracks = []
        unconfirmed_tracks = []

        for i, t in enumerate(self.tracks):
            if t.is_confirmed():
                confirmed_tracks.append(i)
            else:
                unconfirmed_tracks.append(i)

        # Сопоставляем подтвержденные треки
        matches_a, unmatched_tracks_a, unmatched_detections = self._linear_assignment(
            confirmed_tracks, detections)

        # Сопоставляем оставшиеся неподтвержденные треки
        matches_b, unmatched_tracks_b, unmatched_detections = self._linear_assignment(
            unconfirmed_tracks, detections, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = unmatched_tracks_a + unmatched_tracks_b

        return matches, unmatched_tracks, unmatched_detections

    def _linear_assignment(self, track_indices, detections, unmatched_detections=None):
        if unmatched_detections is None:
            unmatched_detections = list(range(len(detections)))

        if len(track_indices) == 0 or len(unmatched_detections) == 0:
            return [], track_indices, unmatched_detections

        features = [detections[i].feature for i in unmatched_detections]
        targets = [self.tracks[i].track_id for i in track_indices]

        cost_matrix = self.metric.distance(features, targets)
        cost_matrix[cost_matrix > self.metric.matching_threshold] = 1e+5

        matches, unmatched_tracks, unmatched_detections_new = [], [], []

        try:
            if cost_matrix.size > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                # Создаем множества для быстрого поиска
                matched_rows = set(row_indices)
                matched_cols = set(col_indices)

                # Обрабатываем совпадения
                for row, col in zip(row_indices, col_indices):
                    track_idx = track_indices[row]
                    detection_idx = unmatched_detections[col]

                    # Проверяем индексы на валидность
                    if (row < len(track_indices) and col < len(unmatched_detections) and
                            track_idx < len(self.tracks) and detection_idx < len(detections)):

                        if cost_matrix[row, col] <= self.metric.matching_threshold:
                            matches.append((track_idx, detection_idx))
                        else:
                            unmatched_tracks.append(track_idx)
                            unmatched_detections_new.append(detection_idx)
                    else:
                        print(f"Invalid indices: track_idx={track_idx}, detection_idx={detection_idx}")

                # Несовпавшие треки
                for i, track_idx in enumerate(track_indices):
                    if i not in matched_rows:
                        unmatched_tracks.append(track_idx)

                # Несовпавшие детекции
                for j, detection_idx in enumerate(unmatched_detections):
                    if j not in matched_cols:
                        unmatched_detections_new.append(detection_idx)

        except Exception as e:
            print(f"Error in linear assignment: {e}")
            # В случае ошибки возвращаем все как несовпавшие
            unmatched_tracks = track_indices.copy()
            unmatched_detections_new = unmatched_detections.copy()

        return matches, unmatched_tracks, unmatched_detections_new

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1