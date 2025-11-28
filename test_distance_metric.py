import numpy as np
from deep_sort import NearestNeighborDistanceMetric


def test_distance_metric():
    print("Testing distance metric...")

    metric = NearestNeighborDistanceMetric("cosine", 0.7)

    # Тест 1: Идентичные векторы
    feature1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    feature2 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    distance = metric._cosine_distance([feature1], [feature2])
    print(f"Test 1 - Identical vectors:")
    print(f"  Feature 1: {feature1}")
    print(f"  Feature 2: {feature2}")
    print(f"  Distance: {distance[0, 0]:.6f}")
    print(f"  Should be 0.0: {distance[0, 0] < 0.001}")

    # Тест 2: Противоположные векторы
    feature3 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    feature4 = np.array([-0.1, -0.2, -0.3, -0.4], dtype=np.float32)

    distance2 = metric._cosine_distance([feature3], [feature4])
    print(f"\nTest 2 - Opposite vectors:")
    print(f"  Feature 3: {feature3}")
    print(f"  Feature 4: {feature4}")
    print(f"  Distance: {distance2[0, 0]:.6f}")
    print(f"  Should be ~2.0: {abs(distance2[0, 0] - 2.0) < 0.001}")

    # Тест 3: Ортогональные векторы
    feature5 = np.array([1.0, 0.0], dtype=np.float32)
    feature6 = np.array([0.0, 1.0], dtype=np.float32)

    distance3 = metric._cosine_distance([feature5], [feature6])
    print(f"\nTest 3 - Orthogonal vectors:")
    print(f"  Feature 5: {feature5}")
    print(f"  Feature 6: {feature6}")
    print(f"  Distance: {distance3[0, 0]:.6f}")
    print(f"  Should be ~1.0: {abs(distance3[0, 0] - 1.0) < 0.001}")

    # Тест 4: Нулевые векторы
    feature7 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    feature8 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    distance4 = metric._cosine_distance([feature7], [feature8])
    print(f"\nTest 4 - Zero vectors:")
    print(f"  Feature 7: {feature7}")
    print(f"  Feature 8: {feature8}")
    print(f"  Distance: {distance4[0, 0]:.6f}")
    print(f"  Should be 0.0: {distance4[0, 0] < 0.001}")


if __name__ == '__main__':
    test_distance_metric()