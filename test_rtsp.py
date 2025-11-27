import cv2
import time


def test_rtsp_stream(rtsp_url):
    print(f"Testing RTSP stream: {rtsp_url}")

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("ERROR: Cannot open RTSP stream")
        return False

    print("RTSP stream opened successfully")

    # Читаем несколько кадров
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i + 1}: OK - Shape: {frame.shape}")
            # Показываем кадр
            cv2.imshow('RTSP Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"Frame {i + 1}: FAILED")

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    return True


if __name__ == '__main__':
    rtsp_url = 'rtsp://admin:admin@10.0.0.242:554/live/main'
    test_rtsp_stream(rtsp_url)