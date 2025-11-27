import numpy as np
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import base64
import cv2

db = SQLAlchemy()


class Visitor(db.Model):
    __tablename__ = 'visitors'

    id = db.Column(db.Integer, primary_key=True)
    track_id = db.Column(db.Integer, unique=True, nullable=False)
    first_seen = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    visit_count = db.Column(db.Integer, default=1)
    is_active = db.Column(db.Boolean, default=True)

    # Детекции
    detections = db.relationship('Detection', backref='visitor', lazy=True)
    appearances = db.relationship('Appearance', backref='visitor', lazy=True)


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
    detection_type = db.Column(db.String(20))  # 'face' or 'clothing'
    image_data = db.Column(db.Text)  # base64 encoded image

    def set_image(self, image):
        """Сохраняем изображение в base64"""
        _, buffer = cv2.imencode('.jpg', image)
        self.image_data = base64.b64encode(buffer).decode('utf-8')

    def get_image(self):
        """Получаем изображение из base64"""
        if self.image_data:
            image_bytes = base64.b64decode(self.image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return None


class Appearance(db.Model):
    __tablename__ = 'appearances'

    id = db.Column(db.Integer, primary_key=True)
    visitor_id = db.Column(db.Integer, db.ForeignKey('visitors.id'))
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    duration = db.Column(db.Float)  # in seconds


class Report(db.Model):
    __tablename__ = 'reports'

    id = db.Column(db.Integer, primary_key=True)
    report_type = db.Column(db.String(50))
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    data = db.Column(db.Text)  # JSON data