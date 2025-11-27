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

        # Загрузка модели YOLOv5 через альтернативный метод
        try:
            # Способ 1: Используем ultralytics package
            from ultralytics import YOLO
            self.model = YOLO('yolov5s.pt')
            self.model_type = 'ultralytics'
            print("YOLO model loaded successfully using ultralytics")

        except ImportError:
            try:
                # Способ 2: Используем torch.hub с force_reload
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
                self.model_type = 'torchhub'
                print("YOLO model loaded successfully using torch.hub")

            except Exception as e:
                print(f"Error loading YOLOv5: {e}")
                # Способ 3: Локальная загрузка
                try:
                    import urllib.request
                    model_path = 'yolov5s.pt'
                    if not os.path.exists(model_path):
                        print("Downloading YOLOv5s model...")
                        url = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt'
                        urllib.request.urlretrieve(url, model_path)

                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                    self.model_type = 'local'
                    print("YOLO model loaded from local file")

                except Exception as e2:
                    print(f"Failed to load YOLO model: {e2}")
                    self.model = None
                    self.model_type = None

        if self.model:
            self.model.conf = conf_threshold
            self.model.iou = nms_threshold
            if hasattr(self.model, 'to'):
                self.model.to(self.device)

        # Классы для детекции
        self.person_class_id = 0  # person class in COCO

    def detect(self, image):
        """Детекция людей"""
        if self.model is None:
            print("YOLO model not available")
            return []

        try:
            # Конвертируем BGR в RGB если нужно
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Детекция в зависимости от типа модели
            if self.model_type == 'ultralytics':
                results = self.model(image_rgb)
                boxes = results[0].boxes

                detections = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()

                        if int(cls) == self.person_class_id and conf >= self.conf_threshold:
                            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                            feature = self._extract_feature(image, bbox)

                            detections.append({
                                'bbox': bbox,
                                'confidence': float(conf),
                                'class': int(cls),
                                'feature': feature
                            })

            else:  # torchhub or local
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

        # Вырезаем область
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return np.random.randn(512).astype(np.float32)

        try:
            # Ресайз для фичей
            crop_resized = cv2.resize(crop, (128, 256))

            # Упрощенное извлечение фич
            if len(crop_resized.shape) == 3:
                feature = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            else:
                feature = crop_resized

            feature = cv2.resize(feature, (16, 32))
            feature = feature.flatten()

            # Нормализация
            feature_norm = np.linalg.norm(feature)
            if feature_norm > 0:
                feature = feature / feature_norm
            else:
                feature = np.zeros_like(feature)

            # Добиваем до 512 размерности если нужно
            if len(feature) < 512:
                feature = np.pad(feature, (0, 512 - len(feature)))
            elif len(feature) > 512:
                feature = feature[:512]

        except Exception as e:
            print(f"Error extracting feature: {e}")
            feature = np.random.randn(512).astype(np.float32)

        return feature.astype(np.float32)


class FaceClothingDetector:
    def __init__(self):
        print("Initializing YOLO detector...")
        self.yolo_detector = YOLODetector()

        if self.yolo_detector.model is None:
            print("WARNING: YOLO detector failed to initialize. Using fallback mode.")
        else:
            print("YOLO detector initialized successfully")

    def detect_face_and_clothing(self, image):
        """Детекция людей с разделением на лицо и одежду"""
        if self.yolo_detector.model is None:
            # Fallback: используем простой детектор на основе движения
            return self._fallback_detection(image)

        detections = self.yolo_detector.detect(image)

        face_detections = []
        clothing_detections = []

        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            feature = det['feature']

            x, y, w, h = bbox

            # Простая эвристика: верхняя часть - лицо, нижняя - одежда
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

    def _fallback_detection(self, image):
        """Простой fallback детектор на основе фона"""
        if not hasattr(self, 'fgbg'):
            self.fgbg = cv2.createBackgroundSubtractorMOG2()
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Вычитание фона
        fgmask = self.fgbg.apply(image)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

        # Находим контуры
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        face_detections = []
        clothing_detections = []

        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Фильтр по размеру
                x, y, w, h = cv2.boundingRect(contour)

                bbox = [float(x), float(y), float(w), float(h)]
                feature = np.random.randn(512).astype(np.float32)

                face_bbox = [x, y, w, h // 3]
                clothing_bbox = [x, y + h // 3, w, h * 2 // 3]

                face_detections.append({
                    'bbox': face_bbox,
                    'confidence': 0.5,
                    'feature': feature
                })

                clothing_detections.append({
                    'bbox': clothing_bbox,
                    'confidence': 0.5,
                    'feature': feature
                })

        return face_detections, clothing_detections