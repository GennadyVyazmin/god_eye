import cv2
import numpy as np
import torch


class SimpleDetector:
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Используем OpenCV для детекции движения
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

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
                if area > 1000:  # Фильтр по размеру
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

        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return np.random.randn(512).astype(np.float32)

        try:
            crop_resized = cv2.resize(crop, (128, 256))

            if len(crop_resized.shape) == 3:
                feature = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            else:
                feature = crop_resized

            feature = cv2.resize(feature, (16, 32))
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
    def __init__(self):
        print("Initializing Simple Detector...")
        self.detector = SimpleDetector()
        print("Simple Detector initialized successfully")

    def detect_face_and_clothing(self, image):
        """Детекция с разделением на лицо и одежду"""
        detections = self.detector.detect(image)

        face_detections = []
        clothing_detections = []

        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            feature = det['feature']

            x, y, w, h = bbox

            # Разделяем на лицо и одежду
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