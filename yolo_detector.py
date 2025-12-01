import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import time
import os


class YOLODetector:
    def __init__(self, conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Загрузка модели YOLO
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            self.model_type = 'ultralytics'
            print("YOLO model loaded successfully using ultralytics")

        except ImportError:
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
                self.model_type = 'torchhub'
                print("YOLO model loaded successfully using torch.hub")
            except Exception as e:
                print(f"Error loading YOLOv5: {e}")
                self.model = None
                self.model_type = None

        if self.model:
            if hasattr(self.model, 'conf'):
                self.model.conf = conf_threshold
            if hasattr(self.model, 'iou'):
                self.model.iou = nms_threshold
            if hasattr(self.model, 'to'):
                self.model.to(self.device)

        self.person_class_id = 0
        self.feature_history = []  # Для отладки

    def detect(self, image):
        """Детекция людей"""
        if self.model is None:
            print("YOLO model not available")
            return []

        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            detections = []

            if self.model_type == 'ultralytics':
                results = self.model(image_rgb, verbose=False)

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        conf = box.conf[0].cpu().numpy()

                        if cls == self.person_class_id and conf >= self.conf_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                            # Фильтруем слишком маленькие детекции
                            if bbox[2] > 50 and bbox[3] > 100:
                                feature = self._extract_simple_geometric_feature(image, bbox)
                                detections.append({
                                    'bbox': bbox,
                                    'confidence': float(conf),
                                    'class': cls,
                                    'feature': feature
                                })
                                print(f"  YOLO detection: bbox={[int(x) for x in bbox]}, conf={conf:.3f}")

            else:  # torchhub
                results = self.model(image_rgb)

                if len(results.xyxy[0]) > 0:
                    for detection in results.xyxy[0]:
                        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()

                        if int(cls) == self.person_class_id and conf >= self.conf_threshold:
                            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                            if bbox[2] > 50 and bbox[3] > 100:
                                feature = self._extract_simple_geometric_feature(image, bbox)
                                detections.append({
                                    'bbox': bbox,
                                    'confidence': float(conf),
                                    'class': int(cls),
                                    'feature': feature
                                })

            return detections

        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []

    def _extract_simple_geometric_feature(self, image, bbox):
        """
        ПРОСТЕЙШИЕ геометрические фичи - самые стабильные!
        """
        x, y, w, h = [int(coord) for coord in bbox]

        # Проверка границ
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(10, min(w, w_img - x))
        h = max(20, min(h, h_img - y))

        # ТОЛЬКО геометрические фичи (нормализованные)
        feature = np.array([
            x / w_img,  # относительная позиция X [0, 1]
            y / h_img,  # относительная позиция Y [0, 1]
            (x + w / 2) / w_img,  # центр по X [0, 1]
            (y + h / 2) / h_img,  # центр по Y [0, 1]
            w / w_img,  # относительная ширина [0, 1]
            h / h_img,  # относительная высота [0, 1]
            w / h,  # соотношение сторон
            (w * h) / (w_img * h_img)  # относительная площадь [0, 1]
        ], dtype=np.float32)

        # Добавляем простые цветовые фичи
        try:
            # Вырезаем небольшую область для цвета
            crop = image[y:y + h, x:x + w]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (8, 8))  # Очень маленький
                # Средний цвет в HSV
                hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
                mean_color = np.mean(hsv, axis=(0, 1))
                # Нормализуем
                mean_color = mean_color / np.array([180, 255, 255], dtype=np.float32)
                feature = np.concatenate([feature, mean_color])
        except:
            pass

        # Добиваем до фиксированного размера (16)
        if len(feature) < 16:
            feature = np.pad(feature, (0, 16 - len(feature)))
        elif len(feature) > 16:
            feature = feature[:16]

        # L2 нормализация - ВАЖНО: ДОЛЖНА БЫТЬ 1.0!
        feature_norm = np.linalg.norm(feature)
        if feature_norm > 0:
            feature = feature / feature_norm
        else:
            feature = np.ones(16, dtype=np.float32) / np.sqrt(16)  # равномерный вектор

        # Отладка - ПРОВЕРЯЕМ НОРМУ!
        actual_norm = np.linalg.norm(feature)
        print(
            f"    Geometric feature: shape={feature.shape}, norm={actual_norm:.6f} (should be 1.0), mean={np.mean(feature):.6f}")
        if abs(actual_norm - 1.0) > 0.01:
            print(f"    ⚠️ WARNING: Feature norm is {actual_norm:.6f}, should be 1.0!")

        return feature.astype(np.float32)

    def _extract_color_based_feature(self, image, bbox):
        """
        Альтернатива: фичи на основе цвета (может быть стабильнее)
        """
        x, y, w, h = [int(coord) for coord in bbox]

        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(10, min(w, w_img - x))
        h = max(20, min(h, h_img - y))

        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return np.zeros(32, dtype=np.float32)

        try:
            # Ресайз к маленькому размеру
            crop_resized = cv2.resize(crop, (32, 64))

            # Конвертация в HSV
            hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

            # Разделяем каналы
            h_channel, s_channel, v_channel = cv2.split(hsv)

            # Простые гистограммы
            hist_h = cv2.calcHist([h_channel], [0], None, [8], [0, 180])
            hist_s = cv2.calcHist([s_channel], [0], None, [4], [0, 256])
            hist_v = cv2.calcHist([v_channel], [0], None, [4], [0, 256])

            # Нормализация гистограмм
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()

            # Средние значения
            mean_h = np.mean(h_channel) / 180.0
            mean_s = np.mean(s_channel) / 255.0
            mean_v = np.mean(v_channel) / 255.0

            # Формируем фичу
            feature = np.concatenate([
                hist_h,  # 8
                hist_s,  # 4
                hist_v,  # 4
                [mean_h, mean_s, mean_v],  # 3
                [w / h, w / w_img, h / h_img]  # 3 геометрических
            ])  # Всего: 22 фичи

            # Нормализация
            feature_norm = np.linalg.norm(feature)
            if feature_norm > 0:
                feature = feature / feature_norm

            # Добиваем до 32
            feature = np.pad(feature, (0, 32 - len(feature)))

            print(f"    Color feature: shape={feature.shape}, norm={feature_norm:.3f}")

            return feature.astype(np.float32)

        except Exception as e:
            print(f"Error in color feature extraction: {e}")
            return np.zeros(32, dtype=np.float32)


class SimpleDetector:
    def __init__(self, conf_threshold=0.3):
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500, varThreshold=16)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        print("Simple motion detector initialized")

    def detect(self, image):
        """Детекция движения"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fgmask = self.fgbg.apply(gray)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(area / 5000.0, 1.0)

                    if confidence >= self.conf_threshold:
                        bbox = [float(x), float(y), float(w), float(h)]
                        # Простая геометрическая фича
                        h_img, w_img = image.shape[:2]
                        feature = np.array([
                            x / w_img, y / h_img,
                            w / w_img, h / h_img,
                            w / h
                        ], dtype=np.float32)
                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class': 0,
                            'feature': feature
                        })

            return detections

        except Exception as e:
            print(f"Error in motion detection: {e}")
            return []


class FaceClothingDetector:
    def __init__(self, use_yolo=True):
        print("Initializing FaceClothingDetector...")

        if use_yolo:
            try:
                self.detector = YOLODetector(conf_threshold=0.5)
                print("YOLO detector initialized successfully")
                self.detector_type = "yolo"
            except Exception as e:
                print(f"YOLO initialization failed: {e}")
                print("Using motion detection fallback")
                self.detector = SimpleDetector()
                self.detector_type = "motion"
        else:
            self.detector = SimpleDetector()
            self.detector_type = "motion"

        print(f"Using detector: {self.detector_type}")

    def detect_face_and_clothing(self, image):
        """Детекция людей"""
        detections = self.detector.detect(image)

        if len(detections) > 0:
            print(f"Detected {len(detections)} person(s)")

        return detections, []