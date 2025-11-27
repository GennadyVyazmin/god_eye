import cv2
import threading
import time
from enum import Enum


class VideoSourceType(Enum):
    CAMERA = "camera"
    RTSP = "rtsp"
    FILE = "file"
    HTTP = "http"


class VideoSource:
    def __init__(self, source, source_type=VideoSourceType.CAMERA):
        self.source = source
        self.source_type = source_type
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        """Запуск захвата видео"""
        try:
            if self.source_type == VideoSourceType.CAMERA:
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                raise Exception(f"Could not open video source: {self.source}")

            self.running = True
            self.thread = threading.Thread(target=self._update_frame, daemon=True)
            self.thread.start()
            print(f"Video source started: {self.source_type.value}://{self.source}")
            return True

        except Exception as e:
            print(f"Error starting video source: {e}")
            return False

    def _update_frame(self):
        """Обновление кадров в отдельном потоке"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("Failed to read frame from video source")
                time.sleep(0.1)

    def get_frame(self):
        """Получение текущего кадра"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Остановка захвата видео"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print("Video source stopped")


class VideoSourceManager:
    def __init__(self):
        self.sources = {}
        self.active_source = None

    def add_source(self, name, source, source_type=VideoSourceType.CAMERA):
        """Добавление видео источника"""
        video_source = VideoSource(source, source_type)
        self.sources[name] = video_source
        return video_source

    def start_source(self, name):
        """Запуск конкретного источника"""
        if name in self.sources:
            if self.active_source and self.active_source != name:
                self.stop_source(self.active_source)

            if self.sources[name].start():
                self.active_source = name
                return True
        return False

    def stop_source(self, name):
        """Остановка источника"""
        if name in self.sources:
            self.sources[name].stop()
            if self.active_source == name:
                self.active_source = None

    def get_active_frame(self):
        """Получение кадра от активного источника"""
        if self.active_source and self.active_source in self.sources:
            return self.sources[self.active_source].get_frame()
        return None

    def get_source_info(self, name):
        """Получение информации об источнике"""
        if name in self.sources and self.sources[name].cap:
            cap = self.sources[name].cap
            return {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            }
        return {}