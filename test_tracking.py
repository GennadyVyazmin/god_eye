import cv2
import numpy as np
from yolo_detector import YOLODetector


def test_feature_consistency():
    detector = YOLODetector()

    # Создаем тестовое изображение
    image1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image1, (100, 100), (200, 300), (255, 255, 255), -1)

    image2 = image1.copy()  # То же самое изображение

    # Детектируем на обоих изображениях
    detections1 = detector.detect(image1)
    detections2 = detector.detect(image2)

    print(f"Detections 1: {len(detections1)}")
    print(f"Detections 2: {len(detections2)}")

    if len(detections1) > 0 and len(detections2) > 0:
        feature1 = detections1[0]['feature']
        feature2 = detections2[0]['feature']

        # Вычисляем косинусное расстояние
        similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        distance = 1 - similarity

        print(f"Feature similarity: {similarity:.4f}")
        print(f"Feature distance: {distance:.4f}")
        print(f"Should match: {distance < 0.2}")


if __name__ == '__main__':
    test_feature_consistency()