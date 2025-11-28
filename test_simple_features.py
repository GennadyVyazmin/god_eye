import cv2
import numpy as np
from simple_detector import SimpleFeatureDetector


def test_simple_features():
    detector = SimpleFeatureDetector()

    print("Testing simple feature extraction...")

    # Создаем тестовое изображение с человеком
    image = detector.create_test_person_image()

    # Сохраняем для просмотра
    cv2.imwrite('test_person.jpg', image)
    print("Saved test image as 'test_person.jpg'")

    # Детектируем
    detections = detector.detect(image)

    print(f"Detections: {len(detections)}")

    if len(detections) > 0:
        feature = detections[0]['feature']
        bbox = detections[0]['bbox']

        print(f"BBox: {[int(x) for x in bbox]}")
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

        # Тест на немного разных bbox (имитация движения)
        bbox2 = [bbox[0] + 5, bbox[1] + 5, bbox[2], bbox[3]]  # немного сдвинули
        feature3 = detector._extract_simple_feature(bbox2, image.shape)
        similarity2 = np.dot(feature1, feature3)
        distance2 = 1 - similarity2

        print(f"Shifted bbox similarity: {similarity2:.4f}")
        print(f"Shifted bbox distance: {distance2:.4f}")
        print(f"Should match (distance < 0.7): {distance2 < 0.7}")

    else:
        print("No detections! Trying fallback test...")
        # Альтернативный тест - создаем искусственный bbox
        test_bbox = [100, 100, 80, 200]
        feature1 = detector._extract_simple_feature(test_bbox, image.shape)
        feature2 = detector._extract_simple_feature(test_bbox, image.shape)

        similarity = np.dot(feature1, feature2)
        distance = 1 - similarity

        print(f"Manual bbox test - Same bbox distance: {distance:.4f}")
        print(f"Should be identical: {distance < 0.001}")


if __name__ == '__main__':
    test_simple_features()