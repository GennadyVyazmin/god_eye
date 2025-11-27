import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import time


class YOLODetector:
    def __init__(self, conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Загрузка модели YOLOv5 через torch.hub
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
            self.model.conf = conf_threshold
            self.model.iou = nms_threshold
            self.model.to(self.device)
            print("YOLOv5 model loaded successfully from torch.hub")
        except Exception as e:
            print(f"Error loading YOLOv5: {e}")
            # Альтернативная загрузка
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', trust_repo=True)
                self.model.to(self.device)
                print("YOLOv5 model loaded from local")
            except Exception as e2:
                print(f"Failed to load YOLO model: {e2}")
                self.model = None

        # Классы для детекции
        self.person_class_id = 0  # person class in COCO

    def detect(self, image):
        """Детекция людей"""
        if self.model is None:
            return []

        # Конвертируем BGR в RGB если нужно
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Детекция
        results = self.model(image_rgb)
        detections = []

        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()

                # Фильтруем только людей
                if int(cls) == self.person_class_id and conf >= self.conf_threshold:
                    # Создаем bounding box [x, y, w, h]
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                    # Извлекаем фичи из bounding box
                    feature = self._extract_feature(image, bbox)

                    detections.append({
                        'bbox': bbox,
                        'confidence': float(conf),
                        'class': int(cls),
                        'feature': feature
                    })

        return detections

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

            # Упрощенное извлечение фич (в реальности используйте reid модель)
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
        self.yolo_detector = YOLODetector()

    def detect_face_and_clothing(self, image):
        """Детекция людей с разделением на лицо и одежду"""
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
                'feature': feature  # Используем ту же фичу для упрощения
            })

            clothing_detections.append({
                'bbox': clothing_bbox,
                'confidence': confidence,
                'feature': feature  # Используем ту же фичу для упрощения
            })

        return face_detections, clothing_detections