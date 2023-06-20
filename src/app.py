import logging
import time

import cv2
import numpy as np

from core.detectors import shapeDetector, colorDetector
from core.normalizers import shapeNormalizer, colorNormalizer
from core.segmentations import by_saturation

logger = logging.Logger('app')
logger.setLevel(logging.DEBUG)

capture: cv2.VideoCapture

WINNAME = 'RESULT'

KEY_CODE_ESC = 27
KEY_CODE_SPACE = ord(' ')
KEY_DELAY_MS = 33


def setup():
    global capture
    capture = cv2.VideoCapture(0)


def teardown():
    capture.release()
    cv2.destroyAllWindows()


def mainloop():
    frame_counts = 0
    start_time = time.time()
    while True:
        key = cv2.waitKey(KEY_DELAY_MS)
        if key == KEY_CODE_ESC:
            break
        if not capture.isOpened():
            logger.warning("카메라를 불러올 수 없습니다.")
            continue
        ret, frame = capture.read()
        if not ret:
            logger.warning("영상을 불러올 수 없습니다.")
            continue
        proc(frame)
        frame_counts += 1
    end_time = time.time()
    duration_sec = end_time-start_time
    logger.info(f'{frame_counts} frames')
    logger.info(f'{duration_sec:.2f} seconds')
    logger.info(f'{frame_counts/duration_sec:.2f} fps')


def proc(raw_frame: np.ndarray):
    frame = shapeNormalizer.normalize(raw_frame)
    frame = colorNormalizer.normalize(frame)

    mask_image = by_saturation(frame)
    pill_image = cv2.copyTo(frame, mask_image)

    color = colorDetector.detect(pill_image)
    shape = shapeDetector.detect(mask_image)

    # Visualizing
    out_frame = np.zeros_like(pill_image)
    out_frame[mask_image == 255] = (255,255,255)
    cv2.drawContours(
        out_frame, [shapeDetector.get_contour()], 0, (0, 0, 255), 2)
    for vertex in shapeDetector.get_vertices():
        cv2.circle(out_frame, vertex[0], radius=3,
                   color=(0, 255, 0), thickness=2)
    out_frame = np.hstack([
        shapeNormalizer.normalize(raw_frame),
        frame,
        out_frame,
    ], dtype=np.uint8)
    out_frame = np.vstack([
        out_frame,
        np.full((32, out_frame.shape[1], 3), colorDetector.get_actual_color()[::-1], dtype=np.uint8),
        np.full((32, out_frame.shape[1], 3), color.value[::-1], dtype=np.uint8),
    ], dtype=np.uint8)
    out_frame = cv2.putText(out_frame, color.name, (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    out_frame = cv2.putText(out_frame, shape.name, (520, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow(WINNAME, out_frame)


def main(debug: bool = False):
    if not debug:
        try:
            setup()
            mainloop()
        except Exception as e:
            logger.error(e)
        finally:
            teardown()
    else:
        setup()
        mainloop()
        teardown()


if __name__ == '__main__':
    main(debug=True)
