#!/usr/bin/env python3
"""
Скрипт запуска сервера видеоаналитики для NVIDIA T400
"""

import argparse
import sys
import os


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

    # Запуск сервера
    from analytics_server import VideoAnalyticsServer
    server = VideoAnalyticsServer(rtsp_url=args.rtsp)

    print(f"Server starting on http://{args.host}:{args.port}")
    print(f"RTSP stream: {args.rtsp}")
    print("Available endpoints:")
    print("  GET / - Main page with video stream")
    print("  GET /api/status - Server status")
    print("  GET /api/video_control - Video stream status")
    print("  GET /api/video_stream - Live video stream with detections")
    print("  GET /api/visitors - List visitors")
    print("  GET /api/statistics - Statistics")

    server.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()