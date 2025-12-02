#!/usr/bin/env python3
"""
Скрипт запуска сервера видеоаналитики для NVIDIA T400
"""

import argparse
import sys
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Video Analytics Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--rtsp', default='rtsp://admin:admin@10.0.0.242:554/live/main', help='RTSP stream URL')

    args = parser.parse_args()

    # Проверка доступности GPU
    import torch
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("Warning: No GPU detected. Using CPU mode.")

    # Импортируем необходимые модули
    from analytics_server import app, socketio, VideoAnalyticsServer
    import analytics_server

    # Создаем сервер с переданным RTSP URL
    server = VideoAnalyticsServer(rtsp_url=args.rtsp)

    # Сохраняем ссылку на сервер в модуле analytics_server для API
    analytics_server.server = server
    analytics_server.server_start_time = datetime.now()

    print(f"Server starting on http://{args.host}:{args.port}")
    print(f"RTSP stream: {args.rtsp}")
    print("Available endpoints:")
    print("  GET / - Main page with WebSocket video stream")
    print("  GET /api/status - Server status")
    print("  GET /api/snapshot - Current snapshot")
    print("  GET /api/visitors - List visitors")
    print("  GET /api/statistics - Statistics")

    # Запускаем RTSP поток
    print("Attempting to start RTSP stream...")
    if not server.start_video_stream():
        print("Warning: Could not start RTSP stream. Server will run with test frame.")

    # Запускаем SocketIO сервер
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()