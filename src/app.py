import logging
import cv2
from core.color import *


KEY_CODE_ESC = 27

logger = logging.Logger('app')
capture: cv2.VideoCapture


def setup():
    global capture
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
    while cv2.waitKey(33) != KEY_CODE_ESC:
        loop()
    capture.release()
    cv2.destroyAllWindows()


def loop():
    if not capture.isOpened():
        logger.warning("카메라를 불러올 수 없습니다.")
        return
    ret, frame = capture.read()
    if not ret:
        return
    cv2.imshow('CAMERA', frame)
    cv2.imshow('COLOR', make_pallete(extract_color(frame)))


if __name__ == '__main__':
    setup()
