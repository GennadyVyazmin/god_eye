import cv2
import sys


def test_rtsp(rtsp_url):
    print(f"Testing RTSP connection to: {rtsp_url}")

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("ERROR: Could not open RTSP stream")
        return False

    print("RTSP stream opened successfully")

    # Пробуем прочитать несколько кадров
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i + 1}: OK - Size: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"Frame {i + 1}: FAILED")

    cap.release()
    return True


if __name__ == '__main__':
    rtsp_url = 'rtsp://admin:admin@10.0.0.242:554/live/main'
    test_rtsp(rtsp_url)