import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import time


class YOLODetector:
    def __init__(self, model_path=None, conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загрузка модели YOLOv5 (легкая версия для T400)
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        except:
            # Альтернативная загрузка
            from yolov5 import YOLOv5
            self.model = YOLOv5('yolov5s.pt')

        self.model.conf = conf_threshold
        self.model.iou = nms_threshold
        self.model.to(self.device)

        # Классы для детекции (лицо и одежда)
        self.face_class_id = 0  # person class in COCO
        self.clothing_classes = [0]  # person class

    def detect(self, image):
        """Детекция лиц и одежды"""
        results = self.model(image)
        detections = []

        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()

                # Фильтруем только людей (класс 0 в COCO)
                if int(cls) == self.face_class_id and conf >= self.conf_threshold:
                    # Создаем bounding box
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    # Извлекаем фичи из bounding box (упрощенно)
                    feature = self._extract_feature(image, bbox)

                    detections.append({
                        'bbox': bbox,
                        'confidence': conf,
                        'class': int(cls),
                        'feature': feature
                    })

        return detections

    def _extract_feature(self, image, bbox):
        """Извлечение фич из области детекции"""
        x, y, w, h = [int(coord) for coord in bbox]

        # Вырезаем область
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return np.random.randn(512)

        # Ресайз для фичей
        crop_resized = cv2.resize(crop, (128, 256))

        # Упрощенное извлечение фич (в реальности используйте reid модель)
        feature = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        feature = cv2.resize(feature, (16, 32))
        feature = feature.flatten()
        feature = feature / np.linalg.norm(feature) if np.linalg.norm(feature) > 0 else feature

        return feature


class FaceClothingDetector:
    def __init__(self):
        self.yolo_detector = YOLODetector()

    def detect_face_and_clothing(self, image):
        """Детекция лиц и одежды"""
        detections = self.yolo_detector.detect(image)

        face_detections = []
        clothing_detections = []

        for det in detections:
            # В YOLO person класс включает и лицо и одежду
            # Для разделения можно использовать дополнительные модели
            bbox = det['bbox']

            # Простая эвристика: верхняя часть - лицо, нижняя - одежда
            x, y, w, h = bbox
            face_bbox = [x, y, w, h // 3]
            clothing_bbox = [x, y + h // 3, w, h * 2 // 3]

            face_detections.append({
                'bbox': face_bbox,
                'confidence': det['confidence'],
                'feature': det['feature']
            })

            clothing_detections.append({
                'bbox': clothing_bbox,
                'confidence': det['confidence'],
                'feature': det['feature']
            })

        return face_detections, clothing_detections