import cv2
import time
import sys


def diagnose_rtsp(rtsp_url):
    print(f"🔍 Diagnosing RTSP stream: {rtsp_url}")

    # Пробуем разные бэкенды
    backends = [
        cv2.CAP_FFMPEG,
        cv2.CAP_GSTREAMER,
        cv2.CAP_ANY
    ]

    backend_names = {
        cv2.CAP_FFMPEG: 'FFMPEG',
        cv2.CAP_GSTREAMER: 'GStreamer',
        cv2.CAP_ANY: 'Any'
    }

    for backend in backends:
        print(f"\n🔄 Trying {backend_names[backend]} backend...")

        cap = cv2.VideoCapture(rtsp_url, backend)

        if not cap.isOpened():
            print(f"❌ {backend_names[backend]} backend failed to open")
            continue

        print(f"✅ {backend_names[backend]} backend opened successfully")

        # Получаем информацию о потоке
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))

        print(f"📊 Stream info: {width}x{height}, {fps} FPS, Codec: {codec}")

        # Пробуем прочитать кадры
        success_count = 0
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                success_count += 1
                print(f"✅ Frame {i + 1}: OK - Shape: {frame.shape}")
            else:
                print(f"❌ Frame {i + 1}: FAILED")

            time.sleep(0.1)

        cap.release()

        if success_count > 0:
            print(f"🎉 SUCCESS: {backend_names[backend]} backend works! Read {success_count}/10 frames")
            return backend

        print(f"❌ {backend_names[backend]} backend failed to read frames")

    return None


if __name__ == '__main__':
    rtsp_url = 'rtsp://admin:admin@10.0.0.242:554/live/main'
    working_backend = diagnose_rtsp(rtsp_url)

    if working_backend:
        print(f"\n💡 Use backend: {working_backend}")
    else:
        print("\n💥 All backends failed!")