import cv2


KEY_CODE_ESC = 27

capture: cv2.VideoCapture


def setup():
    global capture
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    while cv2.waitKey(33) != KEY_CODE_ESC:
        loop()
    capture.release()
    cv2.destroyAllWindows()


def loop():
    ret, frame = capture.read()
    if not ret:
        return
    cv2.imshow('CAMERA', frame)


if __name__ == '__main__':
    setup()