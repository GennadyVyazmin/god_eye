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
                                feature = self._extract_stable_feature(image, bbox)
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
                                feature = self._extract_stable_feature(image, bbox)
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

    def _extract_stable_feature(self, image, bbox):
        """
        УЛУЧШЕННЫЕ фичи для стабильного трекинга
        """
        x, y, w, h = [int(coord) for coord in bbox]

        # Проверка границ
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(10, min(w, w_img - x))
        h = max(20, min(h, h_img - y))

        # Вырезаем область с человеком
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        try:
            # 1. Ресайз с сохранением соотношения сторон
            target_height = 128
            scale = target_height / float(h)
            target_width = int(w * scale)

            if target_width < 10:  # Минимальная ширина
                target_width = 10

            crop_resized = cv2.resize(crop, (target_width, target_height))

            # 2. Конвертация в HSV для цветовых фич
            hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

            # 3. Разделяем каналы
            h_channel, s_channel, v_channel = cv2.split(hsv)

            # 4. Цветовые гистограммы (упрощенные)
            hist_h = cv2.calcHist([h_channel], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([s_channel], [0], None, [8], [0, 256])
            hist_v = cv2.calcHist([v_channel], [0], None, [8], [0, 256])

            # 5. Нормализация гистограмм
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()

            # 6. Текстура через градиенты
            gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

            # 7. Градиенты для текстуры
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx ** 2 + gy ** 2)
            orientation = np.arctan2(gy, gx) * (180 / np.pi)

            # 8. Гистограмма направлений градиентов
            hist_orient, _ = np.histogram(orientation, bins=8, range=(-180, 180), weights=magnitude)
            hist_orient = hist_orient / (np.sum(hist_orient) + 1e-10)

            # 9. Простые статистики
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            mean_magnitude = np.mean(magnitude)
            std_magnitude = np.std(magnitude)

            # 10. Формируем фичи
            color_features = np.concatenate([hist_h, hist_s, hist_v])
            texture_features = np.concatenate(
                [hist_orient, [mean_intensity, std_intensity, mean_magnitude, std_magnitude]])

            # 11. Геометрические фичи
            geometric_features = np.array([
                w / h,  # соотношение сторон
                w / w_img,  # относительная ширина
                h / h_img,  # относительная высота
                (x + w / 2) / w_img,  # относительная позиция X
                (y + h / 2) / h_img  # относительная позиция Y
            ])

            # 12. Объединяем все фичи
            feature = np.concatenate([color_features, texture_features, geometric_features])

            # 13. Нормализация
            feature_norm = np.linalg.norm(feature)
            if feature_norm > 0:
                feature = feature / feature_norm

            # 14. Добиваем до фиксированного размера (128)
            if len(feature) < 128:
                feature = np.pad(feature, (0, 128 - len(feature)))
            elif len(feature) > 128:
                feature = feature[:128]

            return feature.astype(np.float32)

        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(128, dtype=np.float32)


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
                        feature = self._extract_feature(image, bbox)
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

    def _extract_feature(self, image, bbox):
        x, y, w, h = [int(coord) for coord in bbox]
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        try:
            crop_resized = cv2.resize(crop, (32, 64))
            if len(crop_resized.shape) == 3:
                feature_img = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            else:
                feature_img = crop_resized

            feature_img = cv2.normalize(feature_img, None, 0, 255, cv2.NORM_MINMAX)
            feature = feature_img.flatten()

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
            feature = np.zeros(128, dtype=np.float32)

        return feature.astype(np.float32)


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