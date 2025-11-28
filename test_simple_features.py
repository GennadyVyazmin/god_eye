import cv2
import numpy as np
from simple_detector import SimpleFeatureDetector


def test_simple_features():
    detector = SimpleFeatureDetector()

    # Создаем тестовое изображение
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.rectangle(image, (100, 100), (200, 300), (200, 200, 200), -1)

    print("Testing simple feature extraction...")

    # Детектируем
    detections = detector.detect(image)

    print(f"Detections: {len(detections)}")

    if len(detections) > 0:
        feature = detections[0]['feature']
        bbox = detections[0]['bbox']

        print(f"BBox: {bbox}")
        print(f"Feature shape: {feature.shape}")
        print(f"Feature: {feature}")
        print(f"Feature norm: {np.linalg.norm(feature):.4f}")

        # Тест на одинаковых bbox
        feature1 = detector._extract_simple_feature(bbox, image.shape)
        feature2 = detector._extract_simple_feature(bbox, image.shape)

        similarity = np.dot(feature1, feature2)
        distance = 1 - similarity

        print(f"Same bbox similarity: {similarity:.4f}")
        print(f"Same bbox distance: {distance:.4f}")
        print(f"Should be identical: {distance < 0.001}")


if __name__ == '__main__':
    test_simple_features()