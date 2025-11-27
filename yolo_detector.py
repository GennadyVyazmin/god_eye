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
            self.model = YOLO('yolov8n.pt')  # Самая легкая модель для тестирования
            self.model_type = 'ultralytics'
            print("YOLO model loaded successfully using ultralytics")

        except ImportError:
            try:
                # Fallback на torch.hub
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

        # Классы для детекции (COCO dataset)
        self.person_class_id = 0  # person class in COCO

    def detect(self, image):
        """Детекция людей"""
        if self.model is None:
            print("YOLO model not available")
            return []

        try:
            # Конвертируем BGR в RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Детекция в зависимости от типа модели
            if self.model_type == 'ultralytics':
                results = self.model(image_rgb, verbose=False)
                detections = []

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        conf = box.conf[0].cpu().numpy()

                        if cls == self.person_class_id and conf >= self.conf_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                            # Фильтр по размеру - убираем слишком маленькие детекции
                            if bbox[2] > 50 and bbox[3] > 100:  # min width 50px, min height 100px
                                feature = self._extract_feature(image, bbox)

                                detections.append({
                                    'bbox': bbox,
                                    'confidence': float(conf),
                                    'class': cls,
                                    'feature': feature
                                })
                            else:
                                print(f"  Filtered small detection: {bbox}")

            else:  # torchhub
                results = self.model(image_rgb)
                detections = []

                if len(results.xyxy[0]) > 0:
                    for detection in results.xyxy[0]:
                        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()

                        if int(cls) == self.person_class_id and conf >= self.conf_threshold:
                            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                            # Фильтр по размеру
                            if bbox[2] > 50 and bbox[3] > 100:
                                feature = self._extract_feature(image, bbox)

                                detections.append({
                                    'bbox': bbox,
                                    'confidence': float(conf),
                                    'class': int(cls),
                                    'feature': feature
                                })
                            else:
                                print(f"  Filtered small detection: {bbox}")

            return detections

        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []

    def _extract_feature(self, image, bbox):
        """Упрощенное извлечение фич из области детекции"""
        x, y, w, h = [int(coord) for coord in bbox]

        # Вырезаем область с проверкой границ
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return np.random.randn(128).astype(np.float32)

        try:
            # Простые фичи на основе гистограмм
            if len(crop.shape) == 3:
                # Цветовые гистограммы
                hist_b = cv2.calcHist([crop], [0], None, [8], [0, 256]).flatten()
                hist_g = cv2.calcHist([crop], [1], None, [8], [0, 256]).flatten()
                hist_r = cv2.calcHist([crop], [2], None, [8], [0, 256]).flatten()

                # Grayscale гистограмма
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                hist_gray = cv2.calcHist([gray], [0], None, [8], [0, 256]).flatten()

                feature = np.concatenate([hist_b, hist_g, hist_r, hist_gray])
            else:
                feature = cv2.calcHist([crop], [0], None, [32], [0, 256]).flatten()

            # Нормализация
            feature_norm = np.linalg.norm(feature)
            if feature_norm > 0:
                feature = feature / feature_norm

            # Добиваем до 128 размерности
            if len(feature) < 128:
                feature = np.pad(feature, (0, 128 - len(feature)))
            elif len(feature) > 128:
                feature = feature[:128]

        except Exception as e:
            feature = np.random.randn(128).astype(np.float32)

        return feature.astype(np.float32)


class SimpleDetector:
    def __init__(self, conf_threshold=0.3):
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Используем OpenCV для детекции движения
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500, varThreshold=16)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        print("Simple motion detector initialized")

    def detect(self, image):
        """Детекция движения"""
        try:
            # Конвертируем в grayscale для детекции движения
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Вычитание фона
            fgmask = self.fgbg.apply(gray)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

            # Находим контуры
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detections = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Фильтр по размеру
                    x, y, w, h = cv2.boundingRect(contour)

                    # Рассчитываем confidence на основе размера области
                    confidence = min(area / 5000.0, 1.0)

                    if confidence >= self.conf_threshold:
                        bbox = [float(x), float(y), float(w), float(h)]
                        feature = self._extract_feature(image, bbox)

                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class': 0,  # person class
                            'feature': feature
                        })

            return detections

        except Exception as e:
            print(f"Error in motion detection: {e}")
            return []

    def _extract_feature(self, image, bbox):
        """Упрощенное извлечение фич"""
        x, y, w, h = [int(coord) for coord in bbox]

        # Проверка границ
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return np.random.randn(128).astype(np.float32)

        try:
            crop_resized = cv2.resize(crop, (64, 128))

            if len(crop_resized.shape) == 3:
                feature = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            else:
                feature = crop_resized

            feature = cv2.resize(feature, (32, 64))
            feature = feature.flatten()

            feature_norm = np.linalg.norm(feature)
            if feature_norm > 0:
                feature = feature / feature_norm
            else:
                feature = np.zeros_like(feature)

            if len(feature) < 128:
                feature = np.pad(feature, (0, 128 - len(feature)))
            elif len(feature) > 128:
                feature = feature[:128]

        except Exception as e:
            feature = np.random.randn(128).astype(np.float32)

        return feature.astype(np.float32)


class FaceClothingDetector:
    def __init__(self, use_yolo=True):
        print("Initializing FaceClothingDetector...")

        if use_yolo:
            try:
                self.detector = YOLODetector(conf_threshold=0.3)  # Низкий порог для лучшей детекции
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
        """Детекция людей - возвращаем ТОЛЬКО уникальные детекции"""
        detections = self.detector.detect(image)

        # Логируем количество обнаруженных людей
        if len(detections) > 0:
            confidences = [d['confidence'] for d in detections]
            print(f"Detected {len(detections)} person(s) with confidence: {confidences}")

        # Вместо дублирования возвращаем пустой список для clothing
        # Это предотвратит двойные детекции одного и того же человека
        return detections, []  # Только лица, одежда пустая