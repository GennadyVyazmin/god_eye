import numpy as np
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import base64
import cv2
import json

db = SQLAlchemy()


class Visitor(db.Model):
    __tablename__ = 'visitors'

    id = db.Column(db.Integer, primary_key=True)
    track_id = db.Column(db.Integer, nullable=False, index=True)
    unique_visitor_id = db.Column(db.String(64), unique=True, index=True)  # Уникальный ID на 20 часов
    first_seen = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    total_visits = db.Column(db.Integer, default=1)  # Общее количество визитов
    today_visits = db.Column(db.Integer, default=1)  # Визитов сегодня
    is_active = db.Column(db.Boolean, default=True)

    # Храним среднюю фичу для повторной идентификации
    avg_feature = db.Column(db.Text)  # JSON с усредненной фичей
    feature_count = db.Column(db.Integer, default=0)  # Количество фич для усреднения

    # Лучшее фото посетителя
    best_photo_data = db.Column(db.Text)  # base64 лучшего фото
    best_photo_confidence = db.Column(db.Float, default=0.0)

    # Дополнительная информация
    last_known_location = db.Column(db.String(100))  # Последнее известное место
    total_duration = db.Column(db.Float, default=0.0)  # Общее время в секундах

    detections = db.relationship('Detection', backref='visitor', lazy=True)
    appearances = db.relationship('Appearance', backref='visitor', lazy=True)
    photos = db.relationship('VisitorPhoto', backref='visitor', lazy=True)


class VisitorPhoto(db.Model):
    __tablename__ = 'visitor_photos'

    id = db.Column(db.Integer, primary_key=True)
    visitor_id = db.Column(db.Integer, db.ForeignKey('visitors.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_data = db.Column(db.Text)  # base64 encoded image
    confidence = db.Column(db.Float)  # Качество фото (размер лица, четкость)
    bbox_x = db.Column(db.Float)
    bbox_y = db.Column(db.Float)
    bbox_w = db.Column(db.Float)
    bbox_h = db.Column(db.Float)
    feature = db.Column(db.Text)  # JSON фичи на момент фото

    def set_image(self, image):
        """Сохраняем изображение в base64"""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.image_data = base64.b64encode(buffer).decode('utf-8')

    def get_image(self):
        """Получаем изображение из base64"""
        if self.image_data:
            image_bytes = base64.b64decode(self.image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return None


class Detection(db.Model):
    __tablename__ = 'detections'

    id = db.Column(db.Integer, primary_key=True)
    visitor_id = db.Column(db.Integer, db.ForeignKey('visitors.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    bbox_x = db.Column(db.Float)
    bbox_y = db.Column(db.Float)
    bbox_w = db.Column(db.Float)
    bbox_h = db.Column(db.Float)
    confidence = db.Column(db.Float)
    detection_type = db.Column(db.String(20))
    image_data = db.Column(db.Text)
    feature = db.Column(db.Text)  # JSON фичи детекции

    def set_image(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        self.image_data = base64.b64encode(buffer).decode('utf-8')


class Appearance(db.Model):
    __tablename__ = 'appearances'

    id = db.Column(db.Integer, primary_key=True)
    visitor_id = db.Column(db.Integer, db.ForeignKey('visitors.id'))
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    duration = db.Column(db.Float)


class Report(db.Model):
    __tablename__ = 'reports'

    id = db.Column(db.Integer, primary_key=True)
    report_type = db.Column(db.String(50))
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    data = db.Column(db.Text)