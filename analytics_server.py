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

from models import db, Visitor, Detection, Appearance, Report
from yolo_detector import FaceClothingDetector
from deep_sort import Tracker, NearestNeighborDistanceMetric, Detection as DeepSortDetection


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

        # Статистика
        self.active_visitors = {}
        self.visitor_counter = 0
        self.last_processed = None

        print("Video Analytics Server initialized successfully")
        print(f"RTSP URL: {rtsp_url}")

    def setup_database(self):
        """Настройка базы данных"""
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analytics.db'
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(self.app)

        with self.app.app_context():
            db.create_all()

    def setup_routes(self):
        """Настройка API маршрутов"""
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
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # Настройки для RTSP
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.cap.isOpened():
                raise Exception(f"Could not open RTSP stream: {self.rtsp_url}")

            # Проверяем первый кадр
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Could not read frame from RTSP stream")

            print(f"RTSP stream connected successfully. Frame size: {frame.shape[1]}x{frame.shape[0]}")

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
        while self.processing:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                else:
                    print("Failed to read frame from RTSP stream. Reconnecting...")
                    self._reconnect_stream()
                    time.sleep(2)
            except Exception as e:
                print(f"Error reading frame: {e}")
                time.sleep(1)

    def _reconnect_stream(self):
        """Переподключение к RTSP потоку"""
        try:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
            if self.frame is not None:
                try:
                    # Обработка кадра
                    self.process_frame(self.frame)
                    self.last_processed = datetime.now()
                except Exception as e:
                    print(f"Error processing frame: {e}")
            time.sleep(0.033)  # ~30 FPS

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

                # Обновление/создание посетителя в БД
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

                    # Создаем новое появление
                    appearance = Appearance(visitor_id=visitor.id, start_time=now)
                    db.session.add(appearance)

                    self.visitor_counter += 1
                    print(f"New visitor created: track_id={track_id}")
                else:
                    # Обновляем время последнего визита
                    visitor.last_seen = now
                    visitor.visit_count = Visitor.visit_count + 1

                    # Обновляем текущее появление
                    appearance = Appearance.query.filter_by(
                        visitor_id=visitor.id,
                        end_time=None
                    ).first()

                    if not appearance:
                        appearance = Appearance(visitor_id=visitor.id, start_time=now)
                        db.session.add(appearance)

                # Сохраняем детекцию (только каждые 10 кадров для экономии места)
                if track_id % 10 == 0:
                    x, y, w, h = bbox
                    x, y, w, h = int(x), int(y), int(w), int(h)

                    # Проверяем границы
                    if (x >= 0 and y >= 0 and w > 0 and h > 0 and
                            x + w <= frame.shape[1] and y + h <= frame.shape[0]):

                        crop = frame[y:y + h, x:x + w]

                        if crop.size > 0:
                            detection = Detection(
                                visitor_id=visitor.id,
                                bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
                                confidence=1.0,
                                detection_type='person'
                            )
                            detection.set_image(crop)
                            db.session.add(detection)

                db.session.commit()

        except Exception as e:
            print(f"Error updating visitor: {e}")
            db.session.rollback()

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
            print(f"New active visitor: {track_id}")

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
            print(f"Visitor {track_id} marked as inactive")

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
                    'visit_times': [v.first_seen.isoformat() for v in visitors],
                    'average_visit_duration': 0
                }

            elif report_type == 'popular_times':
                hours = [0] * 24
                visitors = Visitor.query.filter(
                    Visitor.first_seen >= start_date,
                    Visitor.first_seen <= end_date
                ).all()

                for visitor in visitors:
                    hour = visitor.first_seen.hour
                    hours[hour] += 1

                data = {
                    'hours': hours,
                    'peak_hour': hours.index(max(hours))
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
        if not self.start_video_stream():
            print("Warning: Could not start RTSP stream. Server will run without video processing.")

        print(f"Starting Video Analytics Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=False)


# API Resources
class VideoControl(Resource):
    def get(self):
        """Статус видео потока"""
        try:
            status = {
                'processing': server.processing,
                'rtsp_url': server.rtsp_url,
                'active_visitors': len(server.active_visitors),
                'last_processed': server.last_processed.isoformat() if server.last_processed else None
            }
            return status, 200
        except Exception as e:
            return {'error': str(e)}, 500

    def post(self):
        """Управление видео потоком"""
        try:
            data = request.get_json()
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
                return {'error': 'Invalid action'}, 400

        except Exception as e:
            return {'error': str(e)}, 500


class VideoStream(Resource):
    def get(self):
        """Потоковое видео с детекциями"""

        def generate():
            while server.processing and server.frame is not None:
                try:
                    frame = server.frame.copy()

                    # Получаем текущие треки
                    tracks = server.process_frame(frame)

                    # Рисуем bounding boxes и ID
                    for track_id, track in tracks.items():
                        x, y, w, h = track['bbox']
                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

                        # Рисуем прямоугольник
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Добавляем ID
                        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Добавляем confidence
                        confidence = track.get('confidence', 0)
                        cv2.putText(frame, f'Conf: {confidence:.2f}', (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Добавляем общую статистику
                    cv2.putText(frame, f'Active Visitors: {len(server.active_visitors)}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f'Total Detected: {server.visitor_counter}',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Кодируем в JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                    time.sleep(0.033)  # ~30 FPS

                except Exception as e:
                    print(f"Error generating stream: {e}")
                    time.sleep(0.1)

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


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

            # Рисуем bounding boxes на изображении
            for track_id, track in tracks.items():
                x, y, w, h = track['bbox']
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Кодируем результат
            _, buffer = cv2.imencode('.jpg', frame)
            result_image = base64.b64encode(buffer).decode('utf-8')

            return {
                'tracks': tracks,
                'visitor_count': len(server.active_visitors),
                'processed_image': result_image
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
            report_type = data.get('report_type')
            start_date = datetime.fromisoformat(data.get('start_date'))
            end_date = datetime.fromisoformat(data.get('end_date'))

            report_id = server.generate_report(report_type, start_date, end_date)

            return {'report_id': report_id}, 200

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
                    'data': json.loads(r.data)
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
                    'rtsp_stream': server.rtsp_url
                }, 200

        except Exception as e:
            return {'error': str(e)}, 500


# Глобальный экземпляр сервера
server = VideoAnalyticsServer()

if __name__ == '__main__':
    server.run()