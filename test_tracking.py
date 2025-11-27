#!/usr/bin/env python3
"""
Простой тест трекинга
"""

import cv2
import numpy as np
from analytics_server import VideoAnalyticsServer


def test_tracking():
    # Создаем тестовый сервер
    server = VideoAnalyticsServer()

    # Создаем тестовый кадр с человеком
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Добавляем "человека" в кадр
    cv2.rectangle(test_frame, (500, 300), (700, 800), (255, 255, 255), -1)

    # Тестируем обработку кадра
    print("Testing frame processing...")
    tracks = server.process_frame(test_frame)

    print(f"Found {len(tracks)} tracks")
    for track_id, track_data in tracks.items():
        print(f"Track {track_id}: {track_data}")

    # Проверяем активных посетителей
    print(f"Active visitors: {len(server.active_visitors)}")


if __name__ == '__main__':
    test_tracking()