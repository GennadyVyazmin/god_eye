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

        # Загрузка модели YOLOv5 через ultralytics
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov5s.pt')  # или 'yolov8s.pt'
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
                            feature = self._extract_feature(image, bbox)

                            detections.append({
                                'bbox': bbox,
                                'confidence': float(conf),
                                'class': cls,
                                'feature': feature
                            })

            else:  # torchhub
                results = self.model(image_rgb)
                detections = []

                if len(results.xyxy[0]) > 0:
                    for detection in results.xyxy[0]:
                        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()

                        if int(cls) == self.person_class_id and conf >= self.conf_threshold:
                            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                            feature = self._extract_feature(image, bbox)

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
            return np.random.randn(512).astype(np.float32)

        try:
            # Ресайз для фич
            crop_resized = cv2.resize(crop, (64, 128))  # Уменьшили размер для скорости

            # Упрощенное извлечение фич (цветовые гистограммы + HOG-like фичи)
            if len(crop_resized.shape) == 3:
                # Цветовые гистограммы по каналам
                hist_b = cv2.calcHist([crop_resized], [0], None, [16], [0, 256]).flatten()
                hist_g = cv2.calcHist([crop_resized], [1], None, [16], [0, 256]).flatten()
                hist_r = cv2.calcHist([crop_resized], [2], None, [16], [0, 256]).flatten()

                # Grayscale для текстурных фич
                gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)

                # Простые текстурные фичи (градиенты)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
                texture_feat = np.histogram(gradient_mag, bins=16)[0]

                # Объединяем все фичи
                feature = np.concatenate([hist_b, hist_g, hist_r, texture_feat])

            else:
                feature = crop_resized.flatten()

            # Нормализация
            feature_norm = np.linalg.norm(feature)
            if feature_norm > 0:
                feature = feature / feature_norm
            else:
                feature = np.zeros_like(feature)

            # Добиваем до нужной размерности
            if len(feature) < 512:
                feature = np.pad(feature, (0, 512 - len(feature)))
            elif len(feature) > 512:
                feature = feature[:512]

        except Exception as e:
            print(f"Error extracting feature: {e}")
            feature = np.random.randn(512).astype(np.float32)

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
            return np.random.randn(512).astype(np.float32)

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

            if len(feature) < 512:
                feature = np.pad(feature, (0, 512 - len(feature)))
            elif len(feature) > 512:
                feature = feature[:512]

        except Exception as e:
            feature = np.random.randn(512).astype(np.float32)

        return feature.astype(np.float32)


class FaceClothingDetector:
    def __init__(self, use_yolo=True):
        print("Initializing FaceClothingDetector...")

        if use_yolo:
            try:
                self.detector = YOLODetector(conf_threshold=0.5)  # Понизим порог для лучшей детекции
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
        """Детекция людей с разделением на лицо и одежду"""
        detections = self.detector.detect(image)

        # Логируем количество обнаруженных людей
        if len(detections) > 0:
            print(f"Detected {len(detections)} person(s) with confidence: {[d['confidence'] for d in detections]}")

        face_detections = []
        clothing_detections = []

        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            feature = det['feature']

            x, y, w, h = bbox

            # Разделяем на лицо (верхняя 1/3) и одежду (нижние 2/3)
            face_bbox = [x, y, w, h // 3]
            clothing_bbox = [x, y + h // 3, w, h * 2 // 3]

            face_detections.append({
                'bbox': face_bbox,
                'confidence': confidence,
                'feature': feature
            })

            clothing_detections.append({
                'bbox': clothing_bbox,
                'confidence': confidence,
                'feature': feature
            })

        return face_detections, clothing_detections

    class PersonDetector:
        def __init__(self, conf_threshold=0.5):
            self.conf_threshold = conf_threshold
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")

            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8s.pt')  # Более точная модель
                self.model_type = 'ultralytics'
                print("YOLOv8 model loaded successfully")

            except Exception as e:
                print(f"YOLOv8 initialization failed: {e}")
                try:
                    self.model = YOLO('yolov5s.pt')
                    print("YOLOv5 model loaded successfully")
                except Exception as e2:
                    print(f"All YOLO initializations failed: {e2}")
                    self.model = None

            if self.model:
                print("Person detector initialized successfully")

            self.person_class_id = 0  # person class in COCO

        def detect(self, image):
            """Детекция только людей"""
            if self.model is None:
                return []

            try:
                # Конвертируем BGR в RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image

                # Детекция
                results = self.model(image_rgb, verbose=False, conf=self.conf_threshold)
                detections = []

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        conf = box.conf[0].cpu().numpy()

                        if cls == self.person_class_id and conf >= self.conf_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                            feature = self._extract_feature(image, bbox)

                            detections.append({
                                'bbox': bbox,
                                'confidence': float(conf),
                                'class': cls,
                                'feature': feature
                            })

                return detections

            except Exception as e:
                print(f"Error in person detection: {e}")
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
                return np.random.randn(128).astype(np.float32)  # Уменьшили размерность

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