import cv2
import numpy as np
import torch
from datetime import datetime, timedelta
import json
import base64
from flask import Flask, request, jsonify, send_file
from flask_restful import Api, Resource
import threading
import time
import os

from models import db, Visitor, Detection, Appearance, Report
from yolo_detector import FaceClothingDetector
from deep_sort import Tracker, NearestNeighborDistanceMetric, Detection as DeepSortDetection


class VideoAnalyticsServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.api = Api(self.app)
        self.setup_database()
        self.setup_routes()

        # Инициализация детектора и трекера
        print("Initializing FaceClothingDetector...")
        self.detector = FaceClothingDetector()

        print("Initializing DeepSORT tracker...")
        self.metric = NearestNeighborDistanceMetric("cosine", 0.2)
        self.tracker = Tracker(self.metric, max_iou_distance=0.7, max_age=70, n_init=3)

        # Статистика
        self.active_visitors = {}
        self.visitor_counter = 0
        self.processing = False

        print("Video Analytics Server initialized successfully")

    def setup_database(self):
        """Настройка базы данных"""
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analytics.db'
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(self.app)

        with self.app.app_context():
            db.create_all()

    def setup_routes(self):
        """Настройка API маршрутов"""
        self.api.add_resource(VideoFeed, '/api/video_feed')
        self.api.add_resource(Visitors, '/api/visitors')
        self.api.add_resource(Reports, '/api/reports')
        self.api.add_resource(Statistics, '/api/statistics')

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

                # Сохраняем детекцию
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
                    'visit_times': [v.first_seen.isoformat() for v in visitors],
                    'average_visit_duration': 0  # Можно рассчитать
                }

            elif report_type == 'popular_times':
                # Анализ популярного времени
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
        print(f"Starting Video Analytics Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=False)


# API Resources (остаются без изменений)
class VideoFeed(Resource):
    def post(self):
        """Обработка видео потока"""
        try:
            if 'video' not in request.files:
                return {'error': 'No video file'}, 400

            video_file = request.files['video']

            # Здесь должна быть обработка видео потока
            # В реальном приложении используйте OpenCV для чтения потока

            return {'message': 'Video processing started'}, 200

        except Exception as e:
            return {'error': str(e)}, 500


class Visitors(Resource):
    def get(self):
        """Получение списка посетителей"""
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 20, type=int)

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

            server = VideoAnalyticsServer()
            report_id = server.generate_report(report_type, start_date, end_date)

            return {'report_id': report_id}, 200

        except Exception as e:
            return {'error': str(e)}, 500

    def get(self):
        """Получение отчетов"""
        try:
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
            total_visitors = Visitor.query.count()
            active_visitors = Visitor.query.filter_by(is_active=True).count()
            today_visitors = Visitor.query.filter(
                Visitor.first_seen >= datetime.now().date()
            ).count()

            return {
                'total_visitors': total_visitors,
                'active_visitors': active_visitors,
                'today_visitors': today_visitors
            }, 200

        except Exception as e:
            return {'error': str(e)}, 500


if __name__ == '__main__':
    server = VideoAnalyticsServer()
    server.run()