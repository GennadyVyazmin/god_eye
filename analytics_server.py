import cv2
import numpy as np
import torch
from datetime import datetime, timedelta
import json
import base64
from flask import Flask, request, jsonify, Response
from flask_restful import Api, Resource
from flask_socketio import SocketIO, emit
from yolo_detector import FaceClothingDetector
import threading
import time
import os


# Настройки для OpenCV
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

from models import db, Visitor, Detection, Appearance, Report
from yolo_detector import FaceClothingDetector
from deep_sort import Tracker, NearestNeighborDistanceMetric, Detection as DeepSortDetection

# Создаем Flask app и SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'video-analytics-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
api = Api(app)


class VideoAnalyticsServer:
    def __init__(self, rtsp_url='rtsp://admin:admin@10.0.0.242:554/live/main'):
        self.app = app
        self.socketio = socketio
        self.api = api
        self.rtsp_url = rtsp_url
        self.backend_name = "Unknown"

        # Инициализация детектора и трекера
        print("Initializing FaceClothingDetector...")
        self.detector = FaceClothingDetector(use_yolo=True)  # Используем YOLO детектор

        print("Initializing DeepSORT tracker...")
        self.metric = NearestNeighborDistanceMetric("cosine", 0.7)  # Увеличили порог matching_threshold
        self.tracker = Tracker(
            self.metric,
            max_iou_distance=0.9,  # Увеличили max_iou_distance
            max_age=100,  # Увеличили max_age
            n_init=3  # Уменьшили n_init для быстрого подтверждения
        )

        # Видео поток
        self.cap = None
        self.frame = None
        self.processing = False
        self.stream_thread = None
        self.process_thread = None
        self.websocket_thread = None
        self.websocket_active = False
        self.frame_lock = threading.Lock()
        self.stream_info = {}

        # Статистика
        self.active_visitors = {}
        self.visitor_counter = 0
        self.last_processed = None
        self.frames_processed = 0
        self.frames_read = 0
        self.clients_connected = 0

        # Тестовый кадр если RTSP не работает
        self.test_frame = self._create_test_frame()

        self.setup_database()
        self.setup_routes()
        self.setup_socketio_events()

        print("Video Analytics Server initialized successfully")
        print(f"RTSP URL: {rtsp_url}")

    def _create_test_frame(self):
        """Создание тестового кадра если RTSP не работает"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "RTSP STREAM NOT AVAILABLE", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Check RTSP URL and connection", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"URL: {self.rtsp_url}", (80, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame

    def get_stream_info(self):
        return self.stream_info

    def setup_database(self):
        """Настройка базы данных"""
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analytics.db'
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(self.app)

        with self.app.app_context():
            db.create_all()

    def setup_socketio_events(self):
        """Настройка WebSocket событий"""

        @self.socketio.on('connect')
        def handle_connect():
            self.clients_connected += 1
            print(f'Client connected. Total clients: {self.clients_connected}')
            emit('status', {'message': 'Connected to video stream', 'clients': self.clients_connected})

            # Автоматически запускаем стрим при подключении
            self._start_websocket_stream()

        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.clients_connected = max(0, self.clients_connected - 1)
            print(f'Client disconnected. Total clients: {self.clients_connected}')

        @self.socketio.on('start_stream')
        def handle_start_stream():
            print("WebSocket: Start stream requested by client")
            self._start_websocket_stream()

        @self.socketio.on('stop_stream')
        def handle_stop_stream():
            print("WebSocket: Stop stream requested")
            self.websocket_active = False
            emit('status', {'message': 'WebSocket stream stopped'})

    def _start_websocket_stream(self):
        """Запуск WebSocket потока"""
        if not self.websocket_active:
            self.websocket_active = True
            if not self.websocket_thread or not self.websocket_thread.is_alive():
                self.websocket_thread = threading.Thread(target=self._websocket_stream, daemon=True)
                self.websocket_thread.start()
                print("WebSocket stream started")
                self.socketio.emit('status', {'message': 'WebSocket stream started'})

    def _websocket_stream(self):
        """Поток для отправки кадров через WebSocket"""
        print("WebSocket stream thread started")
        frame_count = 0

        while self.websocket_active and self.clients_connected > 0:
            try:
                # Получаем текущий кадр
                frame = self.get_current_frame()

                # Добавляем детекции если есть реальный кадр
                if self.processing and self.frame is not None:
                    try:
                        tracks = self.process_frame(frame)

                        # Рисуем bounding boxes и ID
                        for track_id, track in tracks.items():
                            x, y, w, h = track['bbox']
                            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

                            # Проверяем границы
                            if (x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]):
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(frame, f'ID: {track_id}', (x1, max(y1 - 10, 20)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error drawing detections: {e}")

                # Добавляем статистику
                status_text = "LIVE" if self.processing and self.frame is not None else "NO SIGNAL"
                status_color = (0, 255, 0) if self.processing and self.frame is not None else (0, 0, 255)

                cv2.putText(frame, f'Status: {status_text}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
                cv2.putText(frame, f'Visitors: {len(self.active_visitors)}', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, f'Frames: {frame_count}', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'WebSocket Stream', (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Ресайзим для оптимизации
                if frame.shape[1] > 800 or frame.shape[0] > 600:
                    frame = cv2.resize(frame, (800, 600))

                # Кодируем в base64
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                if ret:
                    img_base64 = base64.b64encode(buffer).decode('utf-8')

                    # Отправляем через SocketIO
                    self.socketio.emit('video_frame', {
                        'image': f'data:image/jpeg;base64,{img_base64}',
                        'frame_count': frame_count,
                        'timestamp': datetime.now().isoformat(),
                        'status': status_text
                    })
                    frame_count += 1

                    # Логируем каждые 30 кадров
                    if frame_count % 30 == 0:
                        print(f"WebSocket: Sent {frame_count} frames")

                # Пауза для снижения нагрузки (10 FPS)
                time.sleep(0.1)

            except Exception as e:
                print(f"WebSocket stream error: {e}")
                time.sleep(1)

        print("WebSocket stream thread stopped")
        self.websocket_active = False

    def setup_routes(self):
        """Настройка API маршрутов"""

        # Основной маршрут
        @self.app.route('/')
        def index():
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Video Analytics Server</title>
                <meta charset="utf-8">
                <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
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
                    .log { background: #f9f9f9; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; }
                    .controls { margin: 10px 0; }
                    button { padding: 10px 15px; margin: 5px; background: #007cba; color: white; border: none; border-radius: 5px; cursor: pointer; }
                    button:hover { background: #005a87; }
                    .connection-status { padding: 10px; border-radius: 5px; margin: 10px 0; }
                    .connected { background: #d4edda; color: #155724; }
                    .disconnected { background: #f8d7da; color: #721c24; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎥 Video Analytics Server</h1>
                    <p>Сервер видеоаналитики на YOLO + DeepSORT для NVIDIA T400</p>

                    <div class="connection-status" id="connectionStatus">
                        <strong>WebSocket Status:</strong> <span id="wsStatus">Disconnected</span>
                    </div>

                    <div class="controls">
                        <button onclick="connectWebSocket()">🔗 Connect WebSocket</button>
                        <button onclick="startStream()">▶️ Start Stream</button>
                        <button onclick="stopStream()">⏹️ Stop Stream</button>
                        <button onclick="getSnapshot()">📸 Snapshot</button>
                    </div>

                    <div class="stats">
                        <h3>📊 Текущая статистика:</h3>
                        <p><strong>Статус RTSP:</strong> <span id="status">Loading...</span></p>
                        <p><strong>Активные посетители:</strong> <span id="visitors">0</span></p>
                        <p><strong>Всего обнаружено:</strong> <span id="total">0</span></p>
                        <p><strong>Кадр доступен:</strong> <span id="frame">No</span></p>
                        <p><strong>Обработано кадров:</strong> <span id="frames">0</span></p>
                        <p><strong>Подключенные клиенты:</strong> <span id="clients">0</span></p>
                        <p><strong>Бэкенд:</strong> <span id="backend">Unknown</span></p>
                        <p><strong>Разрешение:</strong> <span id="resolution">N/A</span></p>
                        <p><strong>FPS:</strong> <span id="fps">N/A</span></p>
                        <p><strong>RTSP URL:</strong> <code>rtsp://admin:admin@10.0.0.242:554/live/main</code></p>
                    </div>

                    <div class="video-container">
                        <h3>📹 Live Video Stream:</h3>
                        <img id="videoStream" class="video-frame" width="800" height="600" alt="Video Stream" 
                             onerror="this.onerror=null; this.src='/api/snapshot';">
                        <div id="streamInfo">
                            <p>Waiting for video stream...</p>
                        </div>
                    </div>

                    <div class="log-container">
                        <h3>📋 Лог системы:</h3>
                        <div class="log" id="log">Запуск системы...</div>
                    </div>

                    <h2>🔧 Доступные endpoints:</h2>
                    <div class="endpoint">
                        <strong>GET /api/status</strong> - Статус сервера
                    </div>
                    <div class="endpoint">
                        <strong>GET /api/snapshot</strong> - Текущий снимок
                    </div>
                    <div class="endpoint">
                        <strong>GET /api/visitors</strong> - Список посетителей
                    </div>
                    <div class="endpoint">
                        <strong>GET /api/statistics</strong> - Статистика
                    </div>
                </div>

                <script>
                    const socket = io();
                    let frameCount = 0;
                    let isConnected = false;

                    // WebSocket события
                    socket.on('connect', function(data) {
                        isConnected = true;
                        document.getElementById('connectionStatus').className = 'connection-status connected';
                        document.getElementById('wsStatus').textContent = 'Connected';
                        addLog('WebSocket connected successfully');
                        if (data.clients) {
                            document.getElementById('clients').textContent = data.clients;
                        }
                    });

                    socket.on('disconnect', function() {
                        isConnected = false;
                        document.getElementById('connectionStatus').className = 'connection-status disconnected';
                        document.getElementById('wsStatus').textContent = 'Disconnected';
                        addLog('WebSocket disconnected');
                    });

                    socket.on('status', function(data) {
                        addLog('Server: ' + data.message);
                        if (data.clients) {
                            document.getElementById('clients').textContent = data.clients;
                        }
                    });

                    socket.on('video_frame', function(data) {
                        frameCount++;
                        const videoElement = document.getElementById('videoStream');
                        videoElement.src = data.image;
                        document.getElementById('streamInfo').innerHTML = 
                            `<p>Frames received: ${frameCount}, Status: ${data.status}, Last update: ${new Date().toLocaleTimeString()}</p>`;
                    });

                    // Функции управления
                    function connectWebSocket() {
                        if (!isConnected) {
                            socket.connect();
                            addLog('Manual WebSocket connection requested');
                        } else {
                            addLog('WebSocket already connected');
                        }
                    }

                    function startStream() {
                        if (isConnected) {
                            socket.emit('start_stream');
                            addLog('Stream start requested');
                        } else {
                            addLog('Error: WebSocket not connected');
                        }
                    }

                    function stopStream() {
                        if (isConnected) {
                            socket.emit('stop_stream');
                            addLog('Stream stop requested');
                        } else {
                            addLog('Error: WebSocket not connected');
                        }
                    }

                    function getSnapshot() {
                        const timestamp = new Date().getTime();
                        const videoElement = document.getElementById('videoStream');
                        videoElement.src = '/api/snapshot?' + timestamp;
                        addLog('Snapshot loaded');
                    }

                    function updateStatusDisplay(data) {
                        const statusElement = document.getElementById('status');
                        if (data.processing && data.frame_available) {
                            statusElement.innerHTML = '<span class="status-live">🔴 LIVE</span>';
                        } else {
                            statusElement.innerHTML = '<span class="status-off">⚫ NO SIGNAL</span>';
                        }

                        document.getElementById('visitors').textContent = data.active_visitors;
                        document.getElementById('total').textContent = data.total_visitors;
                        document.getElementById('frame').textContent = data.frame_available ? 'Yes' : 'No';
                        document.getElementById('frames').textContent = data.frames_processed || 0;
                        document.getElementById('backend').textContent = data.backend || 'Unknown';

                        if(data.stream_info) {
                            document.getElementById('resolution').textContent = data.stream_info.resolution || 'N/A';
                            document.getElementById('fps').textContent = data.stream_info.fps || 'N/A';
                        }
                    }

                    function addLog(message) {
                        const logElement = document.getElementById('log');
                        const newLog = `[${new Date().toLocaleTimeString()}] ${message}\\n` + logElement.textContent;
                        logElement.textContent = newLog.substring(0, 1000);
                    }

                    // Авто-обновление статуса
                    setInterval(() => {
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(updateStatusDisplay)
                            .catch(error => console.error('Error fetching status:', error));
                    }, 3000);

                    // Запускаем при загрузке
                    window.addEventListener('load', function() {
                        addLog('Page loaded, auto-connecting WebSocket...');
                        // WebSocket автоматически подключится через библиотеку
                    });
                </script>
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
                'frames_read': self.frames_read,
                'clients_connected': self.clients_connected,
                'websocket_active': self.websocket_active,
                'stream_info': self.stream_info,
                'backend': self.backend_name
            })

        @self.app.route('/api/snapshot')
        def snapshot():
            """Получение одного кадра (для тестирования)"""
            try:
                frame = self.get_current_frame()

                # Добавляем информацию о статусе
                status_text = "LIVE" if self.processing and self.frame is not None else "NO SIGNAL"
                status_color = (0, 255, 0) if self.processing and self.frame is not None else (0, 0, 255)

                cv2.putText(frame, f'Status: {status_text}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
                cv2.putText(frame, f'Visitors: {len(self.active_visitors)}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, f'Frames: {self.frames_processed}', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, 'Snapshot', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Ресайзим если нужно
                if frame.shape[1] > 800 or frame.shape[0] > 600:
                    frame = cv2.resize(frame, (800, 600))

                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    response = Response(buffer.tobytes(), mimetype='image/jpeg')
                    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                    response.headers['Pragma'] = 'no-cache'
                    response.headers['Expires'] = '0'
                    return response
                else:
                    return "Error encoding image", 500
            except Exception as e:
                return f"Error: {e}", 500

        # Регистрируем API ресурсы
        self.api.add_resource(Visitors, '/api/visitors')
        self.api.add_resource(Reports, '/api/reports')
        self.api.add_resource(Statistics, '/api/statistics')

    def start_video_stream(self):
        """Запуск RTSP потока"""
        try:
            print(f"Connecting to RTSP stream: {self.rtsp_url}")

            # Пробуем разные бэкенды
            backends = [
                (cv2.CAP_FFMPEG, "FFMPEG"),
                (cv2.CAP_ANY, "ANY")
            ]

            for backend, name in backends:
                print(f"Trying {name} backend...")
                self.cap = cv2.VideoCapture(self.rtsp_url, backend)

                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 15)

                if self.cap.isOpened():
                    self.backend_name = name
                    print(f"{name} backend opened successfully")

                    time.sleep(2)

                    # Пробуем прочитать первый кадр
                    ret, test_frame = self.cap.read()
                    if ret:
                        print(f"Successfully read first frame: {test_frame.shape}")
                        break
                    else:
                        print(f"{name} backend opened but cannot read frames")
                        self.cap.release()
                else:
                    print(f"{name} backend failed to open")

            if not self.cap or not self.cap.isOpened():
                raise Exception("All backends failed to open RTSP stream")

            # Получаем информацию о потоке
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.stream_info = {
                'resolution': f"{width}x{height}",
                'fps': fps,
                'backend': self.backend_name
            }

            print(f"Stream info: {self.stream_info}")

            self.processing = True

            # Запускаем поток для чтения кадров
            self.stream_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.stream_thread.start()

            # Запускаем поток для обработки
            self.process_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.process_thread.start()

            print("Video stream processing started")
            return True

        except Exception as e:
            print(f"Error starting video stream: {e}")
            return False

    def _read_frames(self):
        """Чтение кадров из RTSP потока"""
        consecutive_errors = 0
        max_errors = 5
        success_count = 0

        while self.processing and consecutive_errors < max_errors:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.frame = frame
                    consecutive_errors = 0
                    success_count += 1
                    self.frames_read += 1

                    if success_count % 30 == 0:
                        print(f"Read {success_count} frames from RTSP stream")

                else:
                    consecutive_errors += 1
                    print(f"Failed to read frame ({consecutive_errors}/{max_errors})")

                    if consecutive_errors >= max_errors:
                        print("Too many consecutive errors, stopping stream...")
                        self.processing = False
                        break

                    time.sleep(0.5)

            except Exception as e:
                consecutive_errors += 1
                print(f"Error reading frame: {e}")
                time.sleep(1)

        if consecutive_errors >= max_errors:
            print("RTSP stream stopped due to errors")

    def _processing_loop(self):
        """Основной цикл обработки видео"""
        while self.processing:
            try:
                current_frame = None
                with self.frame_lock:
                    if self.frame is not None:
                        current_frame = self.frame.copy()

                if current_frame is not None:
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
        self.websocket_active = False

        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        if self.websocket_thread:
            self.websocket_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

        print("Video stream stopped")

    def get_current_frame(self):
        """Получение текущего кадра"""
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

            # Логируем общее количество детекций
            total_detections = len(face_detections) + len(clothing_detections)
            if total_detections > 0:
                print(
                    f"Frame {self.frames_processed}: Found {total_detections} detections (faces: {len(face_detections)}, clothing: {len(clothing_detections)})")

            # Объединяем все детекции
            all_detections = face_detections + clothing_detections

            # Конвертация в формат DeepSORT
            deepsort_detections = []
            for i, det in enumerate(all_detections):
                bbox = det['bbox']
                confidence = det['confidence']
                feature = det['feature']

                deepsort_det = DeepSortDetection(bbox, confidence, feature)
                deepsort_detections.append(deepsort_det)
                print(f"  Detection {i}: bbox={bbox}, conf={confidence:.3f}")

            # Логируем состояние трекера до обновления
            print(
                f"Tracks before update: {len(self.tracker.tracks)} (confirmed: {len([t for t in self.tracker.tracks if t.is_confirmed()])})")

            # Обновление трекера
            self.tracker.predict()
            self.tracker.update(deepsort_detections)

            # Логируем состояние трекера после обновления
            confirmed_tracks = [t for t in self.tracker.tracks if t.is_confirmed()]
            print(f"Tracks after update: {len(self.tracker.tracks)} (confirmed: {len(confirmed_tracks)})")

            # Обработка треков
            current_tracks = {}
            for track in confirmed_tracks:
                track_id = track.track_id
                bbox = track.mean[:4].copy()
                bbox[2] *= bbox[3]
                bbox[:2] -= bbox[2:] / 2

                # Убедимся, что координаты валидны
                bbox = [max(0, float(coord)) for coord in bbox]

                current_tracks[track_id] = {
                    'bbox': bbox,
                    'track_id': track_id,
                    'confidence': getattr(track, 'confidence', 1.0),
                    'hits': track.hits
                }

                print(f"  Track {track_id}: bbox={bbox}, hits={track.hits}")

                # Обновление/создание посетителя в БД (только для новых треков)
                if track_id not in self.active_visitors:
                    print(f"  🆕 NEW VISITOR DETECTED: track_id={track_id}")
                    self.update_visitor(track_id, bbox, frame)

            # Обновление активных посетителей
            self.update_active_visitors(current_tracks)

            # Логируем активные треки
            if len(current_tracks) > 0:
                print(f"Active tracks: {list(current_tracks.keys())}")
            else:
                print("No active tracks")

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

        new_visitors = current_ids - previous_ids
        for track_id in new_visitors:
            self.active_visitors[track_id] = {
                'first_seen': datetime.utcnow(),
                'last_seen': datetime.utcnow()
            }

        for track_id in current_ids:
            if track_id in self.active_visitors:
                self.active_visitors[track_id]['last_seen'] = datetime.utcnow()

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
        print("Attempting to start RTSP stream...")
        if not self.start_video_stream():
            print("Warning: Could not start RTSP stream. Server will run with test frame.")

        print(f"Starting Video Analytics Server on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


# API Resources классы
class Visitors(Resource):
    def get(self):
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 20, type=int)

            with app.app_context():
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
        try:
            with app.app_context():
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
        try:
            with app.app_context():
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
                    'stream_info': server.get_stream_info(),
                    'frames_processed': server.frames_processed,
                    'frames_read': server.frames_read,
                    'websocket_active': server.websocket_active,
                    'clients_connected': server.clients_connected
                }, 200

        except Exception as e:
            return {'error': str(e)}, 500


# Глобальный экземпляр сервера
server = VideoAnalyticsServer()
server_start_time = datetime.now()

if __name__ == '__main__':
    server.run()