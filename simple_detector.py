import cv2
import numpy as np
import torch


class SimpleFeatureDetector:
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Загрузка YOLO модели
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            print("YOLO model loaded successfully")
        except ImportError:
            print("Ultralytics not available, using fallback")
            self.model = None

        self.person_class_id = 0

    def detect(self, image):
        """Детекция людей с ПРОСТЫМИ фичами"""
        if self.model is None:
            return []

        try:
            # Конвертируем BGR в RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Детекция
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

                        if bbox[2] > 50 and bbox[3] > 100:
                            # ПРОСТАЯ фича - только размер и положение
                            feature = self._extract_simple_feature(bbox, image.shape)
                            detections.append({
                                'bbox': bbox,
                                'confidence': float(conf),
                                'class': cls,
                                'feature': feature
                            })

            return detections

        except Exception as e:
            print(f"Error in detection: {e}")
            return []

    def _extract_simple_feature(self, bbox, image_shape):
        """СУПЕР ПРОСТАЯ фича на основе положения и размера"""
        x, y, w, h = bbox

        # Нормализуем координаты относительно размера изображения
        img_h, img_w = image_shape[:2]
        x_norm = x / img_w
        y_norm = y / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # Соотношение сторон
        aspect_ratio = w / h if h > 0 else 1.0

        # Площадь
        area = (w * h) / (img_w * img_h)

        # Центр bbox
        center_x = x_norm + w_norm / 2
        center_y = y_norm + h_norm / 2

        # Собираем простую фичу
        feature = np.array([
            x_norm, y_norm,  # позиция
            w_norm, h_norm,  # размер
            aspect_ratio,  # соотношение сторон
            area,  # площадь
            center_x, center_y  # центр
        ], dtype=np.float32)

        # Нормализация
        feature_norm = np.linalg.norm(feature)
        if feature_norm > 0:
            feature = feature / feature_norm

        return feature


class FaceClothingDetector:
    def __init__(self):
        print("Initializing Simple Feature Detector...")
        self.detector = SimpleFeatureDetector(conf_threshold=0.5)
        print("Simple detector initialized successfully")

    def detect_face_and_clothing(self, image):
        """Детекция людей"""
        detections = self.detector.detect(image)

        if len(detections) > 0:
            confidences = [d['confidence'] for d in detections]
            print(f"Detected {len(detections)} person(s) with confidence: {confidences}")

        return detections, []