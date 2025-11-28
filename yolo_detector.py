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

                            if bbox[2] > 50 and bbox[3] > 100:
                                feature = self._extract_improved_feature(image, bbox)
                                detections.append({
                                    'bbox': bbox,
                                    'confidence': float(conf),
                                    'class': cls,
                                    'feature': feature
                                })

            else:  # torchhub
                results = self.model(image_rgb)

                if len(results.xyxy[0]) > 0:
                    for detection in results.xyxy[0]:
                        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()

                        if int(cls) == self.person_class_id and conf >= self.conf_threshold:
                            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                            if bbox[2] > 50 and bbox[3] > 100:
                                feature = self._extract_improved_feature(image, bbox)
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

    def _extract_improved_feature(self, image, bbox):
        """Улучшенное извлечение фич для лучшего сопоставления"""
        x, y, w, h = [int(coord) for coord in bbox]

        # Проверка границ
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return np.random.randn(512).astype(np.float32) * 0.1

        try:
            # Ресайзим к стандартному размеру
            crop_resized = cv2.resize(crop, (64, 128))

            # Конвертируем в grayscale
            gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)

            # Вычисляем HOG-like фичи
            hog_features = self._compute_hog(gray)

            # Цветовые гистограммы
            color_features = self._compute_color_histograms(crop_resized)

            # Текстура (LBP-like)
            texture_features = self._compute_texture(gray)

            # Объединяем все фичи
            feature = np.concatenate([hog_features, color_features, texture_features])

            # Нормализация
            feature_norm = np.linalg.norm(feature)
            if feature_norm > 0:
                feature = feature / feature_norm
            else:
                feature = np.zeros_like(feature)

            return feature.astype(np.float32)

        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.random.randn(512).astype(np.float32) * 0.1

    def _compute_hog(self, gray_image):
        """Вычисление HOG-like фич"""
        try:
            # Простой градиентный подход
            gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

            mag, ang = cv2.cartToPolar(gx, gy)

            # Разбиваем на ячейки 8x8
            cell_size = 8
            h, w = gray_image.shape
            n_cells_x = w // cell_size
            n_cells_y = h // cell_size

            orientation_bins = 9
            hist_range = (0, 2 * np.pi)

            hog_features = []

            for i in range(n_cells_y):
                for j in range(n_cells_x):
                    cell_mag = mag[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
                    cell_ang = ang[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]

                    hist, _ = np.histogram(cell_ang, bins=orientation_bins, range=hist_range, weights=cell_mag)
                    hist = hist / (np.linalg.norm(hist) + 1e-8)  # L2 нормализация

                    hog_features.extend(hist)

            return np.array(hog_features)

        except Exception as e:
            print(f"Error computing HOG: {e}")
            return np.zeros(81)  # 9x9 cells

    def _compute_color_histograms(self, image):
        """Цветовые гистограммы"""
        try:
            # Гистограммы для каждого канала
            hist_b = cv2.calcHist([image], [0], None, [16], [0, 256]).flatten()
            hist_g = cv2.calcHist([image], [1], None, [16], [0, 256]).flatten()
            hist_r = cv2.calcHist([image], [2], None, [16], [0, 256]).flatten()

            # Нормализация
            hist_b = hist_b / (np.sum(hist_b) + 1e-8)
            hist_g = hist_g / (np.sum(hist_g) + 1e-8)
            hist_r = hist_r / (np.sum(hist_r) + 1e-8)

            return np.concatenate([hist_b, hist_g, hist_r])

        except Exception as e:
            print(f"Error computing color histograms: {e}")
            return np.zeros(48)

    def _compute_texture(self, gray_image):
        """Текстурные фичи"""
        try:
            # Простые статистики текстуры
            mean = np.mean(gray_image)
            std = np.std(gray_image)
            entropy = self._compute_entropy(gray_image)

            # Гистограмма градиентов
            gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            grad_hist, _ = np.histogram(mag, bins=16, range=[0, 256])
            grad_hist = grad_hist / (np.sum(grad_hist) + 1e-8)

            texture_features = np.concatenate([[mean, std, entropy], grad_hist])
            return texture_features

        except Exception as e:
            print(f"Error computing texture: {e}")
            return np.zeros(19)

    def _compute_entropy(self, image):
        """Вычисление энтропии изображения"""
        hist, _ = np.histogram(image, bins=256, range=[0, 256])
        hist = hist[hist > 0]
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy


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
            return np.random.randn(512).astype(np.float32) * 0.1

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
            feature = np.random.randn(512).astype(np.float32) * 0.1

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
            confidences = [d['confidence'] for d in detections]
            print(f"Detected {len(detections)} person(s) with confidence: {confidences}")

        return detections, []