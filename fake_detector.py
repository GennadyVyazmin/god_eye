import cv2
import numpy as np


class FakeDetector:
    def __init__(self):
        print("Initializing Fake Detector for testing...")
        self.detection_count = 0

    def detect_face_and_clothing(self, image):
        """Фейковая детекция - всегда возвращает одного человека"""
        self.detection_count += 1

        # Создаем фиксированный bbox
        h, w = image.shape[:2]
        bbox = [w * 0.3, h * 0.2, w * 0.4, h * 0.6]  # 30% от ширины, 20% от высоты

        # Простая фича на основе bbox
        feature = self._extract_simple_feature(bbox, image.shape)

        detection = {
            'bbox': bbox,
            'confidence': 0.8,
            'class': 0,
            'feature': feature
        }

        print(f"Fake detection #{self.detection_count}: bbox={[int(x) for x in bbox]}")

        return [detection], []

    def _extract_simple_feature(self, bbox, image_shape):
        """Такая же простая фича как в simple_detector"""
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