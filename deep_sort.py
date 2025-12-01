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
        print(f"    Track {self.track_id} updated: hits={self.hits}, state={self.state}")

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

    def __repr__(self):
        return f"Track(id={self.track_id}, state={self.state}, hits={self.hits}, age={self.age})"


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
    def __init__(self, metric, matching_threshold, budget=100):
        if metric == "cosine":
            self._metric = self._cosine_distance
        elif metric == "euclidean":
            self._metric = self._euclidean_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}  # track_id -> [features]

    def _cosine_distance(self, x, y):
        """Косинусное расстояние между двумя векторами"""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Если векторы 1D, делаем их 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        # Нормализация
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)

        # Избегаем деления на ноль
        x_norm[x_norm == 0] = 1e-10
        y_norm[y_norm == 0] = 1e-10

        x_normalized = x / x_norm
        y_normalized = y / y_norm

        # Косинусное сходство и расстояние
        cosine_similarity = np.dot(x_normalized, y_normalized.T)
        cosine_distance = 1.0 - cosine_similarity

        return cosine_distance[0, 0] if cosine_distance.shape == (1, 1) else cosine_distance

    def _euclidean_distance(self, x, y):
        """Евклидово расстояние между двумя векторами"""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        # Евклидово расстояние
        dist = np.sqrt(np.sum((x - y) ** 2, axis=1))

        return dist[0] if dist.shape == (1,) else dist

    def partial_fit(self, features, targets, active_targets):
        """Добавляет фичи для указанных треков"""
        for feature, target in zip(features, targets):
            if target not in self.samples:
                self.samples[target] = []
            self.samples[target].append(feature)

            # Ограничиваем количество хранимых фич
            if self.budget is not None and len(self.samples[target]) > self.budget:
                self.samples[target] = self.samples[target][-self.budget:]

        # Удаляем фичи неактивных треков
        self.samples = {k: v for k, v in self.samples.items() if k in active_targets}

    def distance(self, features, targets):
        """
        Вычисление матрицы расстояний между фичами детекций и треков
        """
        cost_matrix = np.ones((len(features), len(targets)), dtype=np.float32) * 1e+5

        if len(features) == 0 or len(targets) == 0:
            return cost_matrix

        for i, feature in enumerate(features):
            for j, target in enumerate(targets):
                if target in self.samples and len(self.samples[target]) > 0:
                    # Используем среднее из последних 3 фич трека
                    track_features = self.samples[target][-3:]
                    distances = []
                    for track_feature in track_features:
                        dist = self._metric(feature, track_feature)
                        distances.append(dist)

                    # Используем минимальное расстояние
                    min_dist = np.min(distances) if distances else 1.0
                    cost_matrix[i, j] = min_dist

                    # DEBUG
                    if min_dist < 0.5:
                        print(f"      Distance Detection {i} -> Track {target}: {min_dist:.3f}")
                else:
                    # Новый трек или трек без фич
                    cost_matrix[i, j] = 0.5  # Среднее расстояние для новых треков

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
        self.frame_count = 0

    def predict(self):
        """Прогноз положения всех треков"""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Обновление треков с новыми детекциями"""
        try:
            self.frame_count += 1
            print(f"\n  [Tracker] Frame {self.frame_count}, Detections: {len(detections)}")

            if not detections:
                # Нет детекций - помечаем все треки как пропущенные
                for track in self.tracks:
                    track.mark_missed()
                self.tracks = [t for t in self.tracks if not t.is_deleted()]
                return [], [], []

            # 1. Получаем активные треки
            confirmed_tracks = []
            tentative_tracks = []

            for i, track in enumerate(self.tracks):
                if track.is_confirmed():
                    confirmed_tracks.append(i)
                elif track.is_tentative():
                    tentative_tracks.append(i)

            print(f"  Active tracks: confirmed={len(confirmed_tracks)}, tentative={len(tentative_tracks)}")

            # 2. Собираем фичи активных треков для метрики
            active_targets = []
            track_features = []
            track_ids = []

            for track_idx in confirmed_tracks + tentative_tracks:
                track = self.tracks[track_idx]
                if track.features:
                    active_targets.append(track.track_id)
                    track_features.append(track.features[-1])  # Последняя фича
                    track_ids.append(track.track_id)

            # 3. Собираем фичи детекций
            detection_features = [d.feature for d in detections]
            detection_indices = list(range(len(detections)))

            # 4. Обновляем метрику с фичами треков
            if track_features:
                self.metric.partial_fit(track_features, track_ids, active_targets)

            # 5. Сопоставление детекций с треками
            matches, unmatched_tracks, unmatched_detections = [], [], []

            if track_ids and detection_features:
                # Сначала сопоставляем подтвержденные треки
                if confirmed_tracks:
                    confirmed_matches, confirmed_unmatched_tracks, unmatched_detections = self._match_tracks(
                        confirmed_tracks, detection_indices, detections)
                    matches.extend(confirmed_matches)
                    unmatched_tracks.extend(confirmed_unmatched_tracks)

                # Затем неподтвержденные треки с оставшимися детекциями
                if tentative_tracks and unmatched_detections:
                    tentative_matches, tentative_unmatched_tracks, unmatched_detections = self._match_tracks(
                        tentative_tracks, unmatched_detections, detections)
                    matches.extend(tentative_matches)
                    unmatched_tracks.extend(tentative_unmatched_tracks)
            else:
                # Нет активных треков - все детекции новые
                unmatched_detections = detection_indices
                unmatched_tracks = confirmed_tracks + tentative_tracks

            # 6. Обновляем совпавшие треки
            for track_idx, detection_idx in matches:
                if 0 <= track_idx < len(self.tracks) and 0 <= detection_idx < len(detections):
                    self.tracks[track_idx].update(self.kf, detections[detection_idx])
                    # Обновляем фичу в метрике
                    self.metric.partial_fit(
                        [detections[detection_idx].feature],
                        [self.tracks[track_idx].track_id],
                        [self.tracks[track_idx].track_id]
                    )

            # 7. Помечаем пропущенные треки
            for track_idx in unmatched_tracks:
                if 0 <= track_idx < len(self.tracks):
                    self.tracks[track_idx].mark_missed()

            # 8. Создаем новые треки из несовпавших детекций
            for detection_idx in unmatched_detections:
                if 0 <= detection_idx < len(detections):
                    self._initiate_track(detections[detection_idx])

            # 9. Удаляем неактивные треки
            self.tracks = [t for t in self.tracks if not t.is_deleted()]

            # 10. Возвращаем результат
            confirmed_count = len([t for t in self.tracks if t.is_confirmed()])
            tentative_count = len([t for t in self.tracks if t.is_tentative()])
            print(
                f"  Tracks after update: total={len(self.tracks)}, confirmed={confirmed_count}, tentative={tentative_count}")

            return matches, unmatched_tracks, unmatched_detections

        except Exception as e:
            print(f"Error in tracker update: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []

    def _match_tracks(self, track_indices, detection_indices, detections):
        """Сопоставление треков и детекций"""
        if not track_indices or not detection_indices:
            return [], track_indices, detection_indices

        # Получаем фичи детекций
        detection_features = [detections[i].feature for i in detection_indices]
        track_ids = [self.tracks[i].track_id for i in track_indices]

        # Вычисляем матрицу расстояний
        cost_matrix = self.metric.distance(detection_features, track_ids)

        print(f"    Matching {len(detection_indices)} detections with {len(track_indices)} tracks")
        print(f"    Cost matrix shape: {cost_matrix.shape}")

        if cost_matrix.size > 0:
            # Применяем порог
            cost_matrix[cost_matrix > self.metric.matching_threshold] = 1e+5

            # Венгерский алгоритм
            matches, unmatched_tracks, unmatched_detections = [], [], []

            if cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                # Обрабатываем совпадения
                for row, col in zip(row_indices, col_indices):
                    if cost_matrix[row, col] <= self.metric.matching_threshold:
                        matches.append((track_indices[col], detection_indices[row]))
                        print(
                            f"    ✅ MATCH: Track {track_ids[col]} -> Detection {detection_indices[row]} (cost: {cost_matrix[row, col]:.3f})")
                    else:
                        unmatched_tracks.append(track_indices[col])
                        unmatched_detections.append(detection_indices[row])

                # Несовпавшие треки
                matched_cols = set(col_indices)
                for j in range(len(track_indices)):
                    if j not in matched_cols:
                        unmatched_tracks.append(track_indices[j])

                # Несовпавшие детекции
                matched_rows = set(row_indices)
                for i in range(len(detection_indices)):
                    if i not in matched_rows:
                        unmatched_detections.append(detection_indices[i])
            else:
                unmatched_tracks = track_indices.copy()
                unmatched_detections = detection_indices.copy()

            return matches, unmatched_tracks, unmatched_detections

        return [], track_indices, detection_indices

    def _initiate_track(self, detection):
        """Создание нового трека из детекции"""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        new_track = Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature
        )
        self.tracks.append(new_track)

        # Немедленно добавляем фичу в метрику
        self.metric.partial_fit(
            [detection.feature],
            [self._next_id],
            [self._next_id]
        )

        print(f"    🆕 NEW TRACK INITIATED: id={self._next_id}, bbox={detection.tlwh}")
        self._next_id += 1