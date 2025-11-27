import cv2
import numpy as np
import torch
from datetime import datetime, timedelta
import json
import base64
from flask import Flask, request, jsonify, Response
from flask_restful import Api, Resource
import threading
import time
import os

# Отключаем GUI бэкенд для OpenCV
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '1'

from models import db, Visitor, Detection, Appearance, Report
from yolo_detector import FaceClothingDetector
from deep_sort import Tracker, NearestNeighborDistanceMetric, Detection as DeepSortDetection


# API Resources классы
class VideoControl(Resource):
    def get(self):
        """Статус видео потока"""
        try:
            status = {
                'processing': server.processing,
                'rtsp_url': server.rtsp_url,
                'active_visitors': len(server.active_visitors),
                'total_visitors': server.visitor_counter,
                'last_processed': server.last_processed.isoformat() if server.last_processed else None,
                'frame_available': server.frame is not None,
                'stream_info': server.get_stream_info()
            }
            return status, 200
        except Exception as e:
            return {'error': str(e)}, 500

    def post(self):
        """Управление видео потоком"""
        try:
            data = request.get_json()
            if not data:
                return {'error': 'No JSON data provided'}, 400

            action = data.get('action')

            if action == 'start':
                if server.start_video_stream():
                    return {'message': 'Video stream started'}, 200
                else:
                    return {'error': 'Failed to start video stream'}, 400

            elif action == 'stop':
                server.stop_video_stream()
                return {'message': 'Video stream stopped'}, 200

            else:
                return {'error': 'Invalid action. Use "start" or "stop"'}, 400

        except Exception as e:
            return {'error': str(e)}, 500


class VideoStream(Resource):
    def get(self):
        """Потоковое видео с детекциями"""

        def generate():
            frame_count = 0
            while True:
                try:
                    # Получаем текущий кадр
                    frame = server.get_current_frame()

                    # Отладочная информация
                    frame_count += 1
                    if frame_count % 100 == 0:  # Логируем каждые 100 кадров
                        print(f"Stream: Sent {frame_count} frames, processing: {server.processing}")

                    # Если есть активная обработка и реальный кадр, добавляем детекции
                    if server.processing and server.frame is not None:
                        tracks = server.process_frame(frame)

                        # Рисуем bounding boxes и ID
                        for track_id, track in tracks.items():
                            x, y, w, h = track['bbox']
                            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

                            # Рисуем прямоугольник
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                            # Добавляем ID
                            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    # Добавляем общую статистику
                    status_text = "LIVE" if server.processing and server.frame is not None else "TEST/NO SIGNAL"
                    cv2.putText(frame, f'Status: {status_text}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(frame, f'Active Visitors: {len(server.active_visitors)}',
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(frame, f'Total Detected: {server.visitor_counter}',
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(frame, f'Frame: {frame_count}',
                                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Ресайзим кадр если он слишком большой для веб-стрима
                    if frame.shape[1] > 1280 or frame.shape[0] > 720:
                        frame = cv2.resize(frame, (1280, 720))

                    # Кодируем в JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        # Fallback на тестовый кадр
                        ret, buffer = cv2.imencode('.jpg', server.test_frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                    time.sleep(0.033)  # ~30 FPS для стрима

                except Exception as e:
                    print(f"Error generating stream: {e}")
                    time.sleep(1)

        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame',
                        headers={
                            'Cache-Control': 'no-cache, no-store, must-revalidate',
                            'Pragma': 'no-cache',
                            'Expires': '0'
                        })


class ProcessImage(Resource):
    def post(self):
        """Обработка единичного изображения"""
        try:
            if 'image' not in request.files:
                return {'error': 'No image file'}, 400

            file = request.files['image']
            img_bytes = file.read()
            img_np = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if frame is None:
                return {'error': 'Invalid image'}, 400

            # Обработка кадра
            tracks = server.process_frame(frame)

            return {
                'tracks': tracks,
                'visitor_count': len(server.active_visitors),
                'total_visitors': server.visitor_counter
            }, 200

        except Exception as e:
            return {'error': str(e)}, 500


class Visitors(Resource):
    def get(self):
        """Получение списка посетителей"""
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 20, type=int)

            with server.app.app_context():
                visitors = Visitor.query.order_by(Visitor.last_seen.desc()).paginate(
                    page=page, per_page=per_page, error_out=False)

                result = {
                    'visitors': [{
                        'id': v.id,
                        'track_id': v.track_id,
                        'first_seen': v.first_seen.isoformat(),
                        'last_seen': v.last_seen.isoformat(),
                        'visit_count': v.visit_count,
                        'is_active': v.is_active
                    } for v in visitors.items],
                    'total': visitors.total,
                    'pages': visitors.pages,
                    'current_page': page
                }

                return result, 200

        except Exception as e:
            return {'error': str(e)}, 500


class Reports(Resource):
    def post(self):
        """Генерация отчета"""
        try:
            data = request.get_json()
            if not data:
                return {'error': 'No JSON data provided'}, 400

            report_type = data.get('report_type')
            start_date_str = data.get('start_date')
            end_date_str = data.get('end_date')

            if not report_type or not start_date_str or not end_date_str:
                return {'error': 'Missing required fields: report_type, start_date, end_date'}, 400

            start_date = datetime.fromisoformat(start_date_str)
            end_date = datetime.fromisoformat(end_date_str)

            report_id = server.generate_report(report_type, start_date, end_date)

            return {'report_id': report_id, 'message': 'Report generated successfully'}, 200

        except Exception as e:
            return {'error': str(e)}, 500

    def get(self):
        """Получение отчетов"""
        try:
            with server.app.app_context():
                reports = Report.query.order_by(Report.generated_at.desc()).all()

                result = [{
                    'id': r.id,
                    'report_type': r.report_type,
                    'generated_at': r.generated_at.isoformat(),
                    'data': json.loads(r.data) if r.data else {}
                } for r in reports]

                return result, 200

        except Exception as e:
            return {'error': str(e)}, 500


class Statistics(Resource):
    def get(self):
        """Получение статистики"""
        try:
            with server.app.app_context():
                total_visitors = Visitor.query.count()
                active_visitors = Visitor.query.filter_by(is_active=True).count()
                today_visitors = Visitor.query.filter(
                    Visitor.first_seen >= datetime.now().date()
                ).count()

                return {
                    'total_visitors': total_visitors,
                    'active_visitors': active_visitors,
                    'today_visitors': today_visitors,
                    'currently_tracking': len(server.active_visitors),
                    'processing_status': server.processing,
                    'rtsp_stream': server.rtsp_url,
                    'server_uptime': str(datetime.now() - server_start_time),
                    'stream_info': server.get_stream_info()
                }, 200

        except Exception as e:
            return {'error': str(e)}, 500


class VideoAnalyticsServer:
    def __init__(self, rtsp_url='rtsp://admin:admin@10.0.0.242:554/live/main'):
        self.app = Flask(__name__)
        self.api = Api(self.app)
        self.setup_database()
        self.setup_routes()

        # RTSP URL
        self.rtsp_url = rtsp_url

        # Инициализация детектора и трекера
        print("Initializing FaceClothingDetector...")
        self.detector = FaceClothingDetector()

        print("Initializing DeepSORT tracker...")
        self.metric = NearestNeighborDistanceMetric("cosine", 0.2)
        self.tracker = Tracker(self.metric, max_iou_distance=0.7, max_age=70, n_init=3)

        # Видео поток
        self.cap = None
        self.frame = None
        self.processing = False
        self.stream_thread = None
        self.process_thread = None
        self.frame_lock = threading.Lock()
        self.stream_info = {}

        # Статистика
        self.active_visitors = {}
        self.visitor_counter = 0
        self.last_processed = None
        self.frames_processed = 0

        # Тестовый кадр если RTSP не работает
        self.test_frame = self._create_test_frame()

        print("Video Analytics Server initialized successfully")
        print(f"RTSP URL: {rtsp_url}")

    def _create_test_frame(self):
        """Создание тестового кадра если RTSP не работает"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "RTSP STREAM NOT AVAILABLE", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Check RTSP URL and connection", (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"URL: {self.rtsp_url}", (30, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame

    def get_stream_info(self):
        """Получение информации о потоке"""
        return self.stream_info

    def setup_database(self):
        """Настройка базы данных"""
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analytics.db'
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(self.app)

        with self.app.app_context():
            db.create_all()

    def setup_routes(self):
        """Настройка API маршрутов"""

        # Основной маршрут
        @self.app.route('/')
        def index():
            return '''
            <html>
                <head>
                    <title>Video Analytics Server</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
                        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                        .video-container { text-align: center; margin: 20px 0; background: #000; padding: 10px; border-radius: 5px; }
                        .video-frame { max-width: 100%; height: auto; border: 2px solid #333; }
                        .stats { background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 15px 0; }
                        .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007cba; }
                        code { background: #eee; padding: 2px 5px; border-radius: 3px; }
                        .status-live { color: green; font-weight: bold; }
                        .status-off { color: red; font-weight: bold; }
                    </style>
                    <script>
                        function updateStatus() {
                            fetch('/api/status')
                                .then(response => response.json())
                                .then(data => {
                                    document.getElementById('status').innerHTML = 
                                        data.processing && data.frame_available ? 
                                        '<span class="status-live">LIVE</span>' : 
                                        '<span class="status-off">OFFLINE</span>';
                                    document.getElementById('visitors').textContent = data.active_visitors;
                                    document.getElementById('total').textContent = data.total_visitors;
                                    document.getElementById('frame').textContent = data.frame_available ? 'Yes' : 'No';
                                    if(data.stream_info) {
                                        document.getElementById('resolution').textContent = data.stream_info.resolution || 'N/A';
                                        document.getElementById('fps').textContent = data.stream_info.fps || 'N/A';
                                    }
                                })
                                .catch(error => {
                                    console.error('Error fetching status:', error);
                                });
                        }

                        // Обновляем статус каждые 5 секунд
                        setInterval(updateStatus, 5000);
                        document.addEventListener('DOMContentLoaded', updateStatus);
                    </script>
                </head>
                <body>
                    <div class="container">
                        <h1>🎥 Video Analytics Server</h1>
                        <p>Сервер видеоаналитики на YOLO + DeepSORT для NVIDIA T400</p>

                        <div class="stats">
                            <h3>📊 Текущая статистика:</h3>
                            <p><strong>Статус:</strong> <span id="status">Loading...</span></p>
                            <p><strong>Активные посетители:</strong> <span id="visitors">0</span></p>
                            <p><strong>Всего обнаружено:</strong> <span id="total">0</span></p>
                            <p><strong>Кадр доступен:</strong> <span id="frame">No</span></p>
                            <p><strong>Разрешение:</strong> <span id="resolution">N/A</span></p>
                            <p><strong>FPS:</strong> <span id="fps">N/A</span></p>
                            <p><strong>RTSP URL:</strong> <code>rtsp://admin:admin@10.0.0.242:554/live/main</code></p>
                        </div>

                        <div class="video-container">
                            <h3>📹 Live Video Stream:</h3>
                            <img src="/api/video_stream" class="video-frame" width="1280" height="720" alt="Video Stream" onerror="this.style.display='none'">
                            <p><a href="/api/video_stream" target="_blank">Открыть в новой вкладке</a></p>
                        </div>

                        <h2>🔧 Доступные endpoints:</h2>

                        <div class="endpoint">
                            <strong>GET /api/status</strong> - Статус сервера
                        </div>

                        <div class="endpoint">
                            <strong>GET /api/video_control</strong> - Статус видео потока
                        </div>

                        <div class="endpoint">
                            <strong>POST /api/video_control</strong> - Управление видео потоком<br>
                            Body: <code>{"action": "start"}</code> или <code>{"action": "stop"}</code>
                        </div>

                        <div class="endpoint">
                            <strong>GET /api/video_stream</strong> - Потоковое видео с детекциями
                        </div>

                        <div class="endpoint">
                            <strong>GET /api/visitors</strong> - Список посетителей
                        </div>

                        <div class="endpoint">
                            <strong>GET /api/statistics</strong> - Статистика
                        </div>

                        <div class="endpoint">
                            <strong>GET /api/reports</strong> - Список отчетов
                        </div>
                    </div>
                </body>
            </html>
            '''

        # API маршруты
        @self.app.route('/api/status')
        def api_status():
            return jsonify({
                'status': 'running',
                'version': '1.0',
                'rtsp_url': self.rtsp_url,
                'processing': self.processing,
                'active_visitors': len(self.active_visitors),
                'total_visitors': self.visitor_counter,
                'last_processed': self.last_processed.isoformat() if self.last_processed else None,
                'frame_available': self.frame is not None,
                'frames_processed': self.frames_processed,
                'stream_info': self.stream_info
            })

        # Регистрируем API ресурсы
        self.api.add_resource(VideoControl, '/api/video_control')
        self.api.add_resource(VideoStream, '/api/video_stream')
        self.api.add_resource(Visitors, '/api/visitors')
        self.api.add_resource(Reports, '/api/reports')
        self.api.add_resource(Statistics, '/api/statistics')
        self.api.add_resource(ProcessImage, '/api/process_image')

    def start_video_stream(self):
        """Запуск RTSP потока"""
        try:
            print(f"Connecting to RTSP stream: {self.rtsp_url}")

            # Используем FFMPEG бэкенд
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            if not self.cap.isOpened():
                print("Trying standard backend...")
                self.cap = cv2.VideoCapture(self.rtsp_url)

            if not self.cap.isOpened():
                raise Exception(f"Could not open RTSP stream: {self.rtsp_url}")

            # Получаем информацию о потоке
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.stream_info = {
                'resolution': f"{width}x{height}",
                'fps': fps,
                'backend': 'FFMPEG' if 'FFMPEG' in str(self.cap.getBackendName()) else 'Standard'
            }

            print(f"RTSP stream connected successfully: {self.stream_info}")

            self.processing = True

            # Запускаем поток для чтения кадров
            self.stream_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.stream_thread.start()

            # Запускаем поток для обработки
            self.process_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.process_thread.start()

            return True

        except Exception as e:
            print(f"Error starting video stream: {e}")
            return False

    def _read_frames(self):
        """Чтение кадров из RTSP потока"""
        error_count = 0
        max_errors = 10
        success_count = 0

        while self.processing and error_count < max_errors:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.frame = frame
                    error_count = 0
                    success_count += 1
                    if success_count % 100 == 0:  # Логируем каждые 100 кадров
                        print(f"RTSP: Successfully read {success_count} frames")
                else:
                    error_count += 1
                    print(f"Failed to read frame from RTSP stream ({error_count}/{max_errors})")
                    if error_count >= max_errors:
                        print("Too many consecutive errors, stopping stream...")
                        self.processing = False
                        break

                    # Пробуем переподключиться после нескольких ошибок
                    if error_count % 3 == 0:
                        self._reconnect_stream()

                    time.sleep(1)

            except Exception as e:
                error_count += 1
                print(f"Error reading frame: {e}")
                time.sleep(1)

    def _reconnect_stream(self):
        """Переподключение к RTSP потоку"""
        try:
            print("Attempting to reconnect to RTSP stream...")
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            if not self.cap.isOpened():
                print("Failed to reconnect to RTSP stream")
                return False

            print("Successfully reconnected to RTSP stream")
            return True

        except Exception as e:
            print(f"Error reconnecting to stream: {e}")
            return False

    def _processing_loop(self):
        """Основной цикл обработки видео"""
        while self.processing:
            try:
                current_frame = None
                with self.frame_lock:
                    if self.frame is not None:
                        current_frame = self.frame.copy()

                if current_frame is not None:
                    # Обработка кадра
                    self.process_frame(current_frame)
                    self.last_processed = datetime.now()
                    self.frames_processed += 1

                time.sleep(0.067)  # ~15 FPS для обработки

            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(1)

    def stop_video_stream(self):
        """Остановка RTSP потока"""
        self.processing = False

        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

        print("Video stream stopped")

    def get_current_frame(self):
        """Получение текущего кадра с блокировкой"""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            else:
                return self.test_frame

    def process_frame(self, frame):
        """Обработка кадра: детекция и трекинг"""
        try:
            # Детекция лиц и одежды
            face_detections, clothing_detections = self.detector.detect_face_and_clothing(frame)

            # Объединяем все детекции
            all_detections = face_detections + clothing_detections

            # Конвертация в формат DeepSORT
            deepsort_detections = []
            for det in all_detections:
                bbox = det['bbox']
                confidence = det['confidence']
                feature = det['feature']

                deepsort_det = DeepSortDetection(bbox, confidence, feature)
                deepsort_detections.append(deepsort_det)

            # Обновление трекера
            self.tracker.predict()
            self.tracker.update(deepsort_detections)

            # Обработка треков
            current_tracks = {}
            for track in self.tracker.tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.mean[:4].copy()
                bbox[2] *= bbox[3]
                bbox[:2] -= bbox[2:] / 2

                # Убедимся, что координаты валидны
                bbox = [max(0, float(coord)) for coord in bbox]

                current_tracks[track_id] = {
                    'bbox': bbox,
                    'track_id': track_id,
                    'confidence': getattr(track, 'confidence', 1.0)
                }

                # Обновление/создание посетителя в БД (только для новых треков)
                if track_id not in self.active_visitors:
                    self.update_visitor(track_id, bbox, frame)

            # Обновление активных посетителей
            self.update_active_visitors(current_tracks)

            return current_tracks

        except Exception as e:
            print(f"Error in process_frame: {e}")
            return {}

    def update_visitor(self, track_id, bbox, frame):
        """Обновление информации о посетителе"""
        try:
            with self.app.app_context():
                visitor = Visitor.query.filter_by(track_id=track_id).first()

                now = datetime.utcnow()

                if not visitor:
                    # Новый посетитель
                    visitor = Visitor(track_id=track_id, first_seen=now, last_seen=now)
                    db.session.add(visitor)
                    db.session.commit()

                    self.visitor_counter += 1
                    print(f"New visitor created: track_id={track_id}")

                db.session.commit()

        except Exception as e:
            print(f"Error updating visitor: {e}")

    def update_active_visitors(self, current_tracks):
        """Обновление списка активных посетителей"""
        current_ids = set(current_tracks.keys())
        previous_ids = set(self.active_visitors.keys())

        # Новые посетители
        new_visitors = current_ids - previous_ids
        for track_id in new_visitors:
            self.active_visitors[track_id] = {
                'first_seen': datetime.utcnow(),
                'last_seen': datetime.utcnow()
            }

        # Обновление времени последнего визита
        for track_id in current_ids:
            if track_id in self.active_visitors:
                self.active_visitors[track_id]['last_seen'] = datetime.utcnow()

        # Удаление неактивных посетителей
        inactive_timeout = timedelta(minutes=5)
        now = datetime.utcnow()
        inactive_visitors = []

        for track_id, data in self.active_visitors.items():
            if track_id not in current_ids:
                if now - data['last_seen'] > inactive_timeout:
                    inactive_visitors.append(track_id)

        for track_id in inactive_visitors:
            del self.active_visitors[track_id]

    def generate_report(self, report_type, start_date, end_date):
        """Генерация отчетов"""
        with self.app.app_context():
            if report_type == 'daily_visitors':
                visitors = Visitor.query.filter(
                    Visitor.first_seen >= start_date,
                    Visitor.first_seen <= end_date
                ).all()

                data = {
                    'total_visitors': len(visitors),
                    'unique_visitors': len(set([v.track_id for v in visitors])),
                    'visit_times': [v.first_seen.isoformat() for v in visitors]
                }

            report = Report(
                report_type=report_type,
                data=json.dumps(data)
            )
            db.session.add(report)
            db.session.commit()

            return report.id

    def run(self, host='0.0.0.0', port=5000):
        """Запуск сервера"""
        # Автоматически запускаем RTSP поток при старте
        print("Attempting to start RTSP stream...")
        if not self.start_video_stream():
            print("Warning: Could not start RTSP stream. Server will run with test frame.")

        print(f"Starting Video Analytics Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=False)


# Глобальный экземпляр сервера
server = VideoAnalyticsServer()
server_start_time = datetime.now()

if __name__ == '__main__':
    server.run()