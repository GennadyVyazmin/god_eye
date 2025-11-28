import cv2
import numpy as np
from fake_detector import FakeDetector


def test_fake_detector():
    detector = FakeDetector()

    # Создаем тестовое изображение
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128

    print("Testing fake detector...")

    # Детектируем несколько раз
    for i in range(3):
        detections, _ = detector.detect_face_and_clothing(image)

        if len(detections) > 0:
            feature = detections[0]['feature']
            bbox = detections[0]['bbox']

            print(f"Detection {i + 1}:")
            print(f"  BBox: {[int(x) for x in bbox]}")
            print(f"  Feature shape: {feature.shape}")
            print(f"  Feature norm: {np.linalg.norm(feature):.4f}")

            # Проверяем консистентность фич
            if i > 0:
                prev_feature = prev_detection['feature']
                similarity = np.dot(prev_feature, feature)
                distance = 1 - similarity
                print(f"  Distance from previous: {distance:.6f}")
                print(f"  Should be identical: {distance < 0.001}")

            prev_detection = detections[0]
        print("-" * 50)


if __name__ == '__main__':
    test_fake_detector()