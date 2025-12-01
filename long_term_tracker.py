import numpy as np
from datetime import datetime, timedelta
import json
from sklearn.metrics.pairwise import cosine_similarity
import hashlib


class LongTermTracker:
    def __init__(self, feature_dim=4, similarity_threshold=0.95, memory_hours=20):
        """
        Долговременный трекер для идентификации людей на 20 часов

        Args:
            feature_dim: размерность фичи
            similarity_threshold: порог похожести для идентификации (0.95 = 95% похожести)
            memory_hours: сколько часов помнить посетителя
        """
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        self.memory_hours = memory_hours

        # Хранилище известных посетителей
        self.known_visitors = {}  # unique_visitor_id -> VisitorData
        self.track_to_visitor = {}  # track_id -> unique_visitor_id

        # Счетчики
        self.next_visitor_id = 1

    def add_visitor(self, track_id, feature, initial_photo=None, bbox=None):
        """Добавление нового посетителя в долговременную память"""
        # Создаем уникальный ID на основе фичи и времени
        visitor_hash = self._create_visitor_hash(feature)
        unique_id = f"VISITOR_{visitor_hash}"

        # Если уже есть похожий посетитель, обновляем его
        existing_id = self._find_similar_visitor(feature)
        if existing_id:
            unique_id = existing_id
            print(f"  🔄 Found similar existing visitor: {unique_id}")
        else:
            print(f"  🆕 New long-term visitor: {unique_id}")

        # Сохраняем данные
        if unique_id not in self.known_visitors:
            self.known_visitors[unique_id] = {
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'features': [feature.tolist()],
                'avg_feature': feature.tolist(),
                'feature_count': 1,
                'total_tracks': 1,
                'current_track': track_id,
                'best_photo': initial_photo,
                'best_photo_bbox': bbox,
                'best_photo_quality': self._calculate_photo_quality(bbox) if bbox else 0
            }
        else:
            # Обновляем существующего
            visitor = self.known_visitors[unique_id]
            visitor['last_seen'] = datetime.now()
            visitor['features'].append(feature.tolist())

            # Обновляем среднюю фичу
            old_avg = np.array(visitor['avg_feature'])
            count = visitor['feature_count']
            new_avg = (old_avg * count + feature) / (count + 1)
            visitor['avg_feature'] = new_avg.tolist()
            visitor['feature_count'] += 1
            visitor['total_tracks'] += 1
            visitor['current_track'] = track_id

            # Обновляем лучшее фото если текущее лучше
            if bbox and initial_photo is not None:
                quality = self._calculate_photo_quality(bbox)
                if quality > visitor['best_photo_quality']:
                    visitor['best_photo'] = initial_photo
                    visitor['best_photo_bbox'] = bbox
                    visitor['best_photo_quality'] = quality
                    print(f"  📸 Updated best photo for {unique_id}, quality: {quality:.2f}")

        # Связываем track_id с unique_id
        self.track_to_visitor[track_id] = unique_id

        return unique_id

    def update_visitor(self, track_id, feature, photo=None, bbox=None):
        """Обновление информации о посетителе"""
        if track_id in self.track_to_visitor:
            unique_id = self.track_to_visitor[track_id]
            visitor = self.known_visitors.get(unique_id)

            if visitor:
                visitor['last_seen'] = datetime.now()
                visitor['features'].append(feature.tolist())

                # Обновляем среднюю фичу
                old_avg = np.array(visitor['avg_feature'])
                count = visitor['feature_count']
                new_avg = (old_avg * count + feature) / (count + 1)
                visitor['avg_feature'] = new_avg.tolist()
                visitor['feature_count'] += 1
                visitor['current_track'] = track_id

                # Обновляем фото если нужно
                if bbox and photo is not None:
                    quality = self._calculate_photo_quality(bbox)
                    if quality > visitor['best_photo_quality'] * 1.1:  # На 10% лучше
                        visitor['best_photo'] = photo
                        visitor['best_photo_bbox'] = bbox
                        visitor['best_photo_quality'] = quality
                        print(f"  📸 New best photo for {unique_id}, quality: {quality:.2f}")

                return unique_id

        return None

    def get_visitor_by_track(self, track_id):
        """Получение уникального ID посетителя по track_id"""
        return self.track_to_visitor.get(track_id)

    def get_active_visitors(self):
        """Получение активных посетителей (были в течение memory_hours)"""
        cutoff_time = datetime.now() - timedelta(hours=self.memory_hours)
        active = {}

        for unique_id, visitor in self.known_visitors.items():
            if visitor['last_seen'] > cutoff_time:
                active[unique_id] = {
                    'unique_id': unique_id,
                    'first_seen': visitor['first_seen'],
                    'last_seen': visitor['last_seen'],
                    'total_tracks': visitor['total_tracks'],
                    'current_track': visitor.get('current_track'),
                    'is_active': visitor.get('current_track') is not None,
                    'best_photo_quality': visitor.get('best_photo_quality', 0)
                }

        return active

    def cleanup_old_visitors(self):
        """Очистка старых посетителей (старше memory_hours)"""
        cutoff_time = datetime.now() - timedelta(hours=self.memory_hours)
        to_remove = []

        for unique_id, visitor in self.known_visitors.items():
            if visitor['last_seen'] < cutoff_time:
                to_remove.append(unique_id)

        for unique_id in to_remove:
            # Удаляем связи с track_id
            tracks_to_remove = [t for t, v in self.track_to_visitor.items() if v == unique_id]
            for track_id in tracks_to_remove:
                del self.track_to_visitor[track_id]

            # Удаляем посетителя
            del self.known_visitors[unique_id]
            print(f"  🗑️ Removed old visitor: {unique_id}")

    def _find_similar_visitor(self, feature):
        """Поиск похожего посетителя по фиче"""
        if not self.known_visitors:
            return None

        feature = feature.reshape(1, -1)
        best_similarity = 0
        best_visitor_id = None

        for unique_id, visitor in self.known_visitors.items():
            # Используем среднюю фичу посетителя
            avg_feature = np.array(visitor['avg_feature']).reshape(1, -1)

            # Вычисляем косинусную схожесть
            similarity = cosine_similarity(feature, avg_feature)[0][0]

            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_visitor_id = unique_id

        if best_visitor_id:
            print(f"  🔍 Found similar visitor {best_visitor_id} with similarity {best_similarity:.3f}")

        return best_visitor_id

    def _create_visitor_hash(self, feature):
        """Создание хэша для уникального ID посетителя"""
        # Округляем фичу для группировки похожих посетителей
        rounded_feature = np.round(feature, 3)
        feature_str = '_'.join([f"{x:.3f}" for x in rounded_feature])

        # Добавляем дату для группировки по дням
        date_str = datetime.now().strftime("%Y%m%d")

        # Создаем хэш
        hash_input = f"{date_str}_{feature_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def _calculate_photo_quality(self, bbox):
        """Оценка качества фото на основе bounding box"""
        if bbox is None:
            return 0

        x, y, w, h = bbox
        # Качество = размер лица + соотношение сторон (близкое к 0.75 - идеальное лицо)
        size_quality = w * h / 10000  # Нормализуем
        aspect_ratio = w / h
        ratio_quality = 1.0 - min(abs(aspect_ratio - 0.75), 0.5) / 0.5

        return size_quality * ratio_quality

    def get_visitor_stats(self):
        """Статистика по посетителям"""
        total = len(self.known_visitors)
        active = len([v for v in self.known_visitors.values() if v.get('current_track')])

        return {
            'total_visitors': total,
            'active_visitors': active,
            'memory_hours': self.memory_hours
        }