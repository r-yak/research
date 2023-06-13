import functools
import logging
import math
import time
import typing

import cv2
import colorgram
import numpy as np
import webcolors
from PIL import Image


KEY_CODE_ESC = 27

logger = logging.Logger('app')
capture: cv2.VideoCapture


CONTOUR_APPROX = 0.04
CIRCLE_MOMENTUM_THRESH = 7


def setup():
    global capture
    start_time = time.time()
    frame_cnt = 0
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
    while cv2.waitKey(33) != KEY_CODE_ESC:
        loop()
        frame_cnt+=1
    capture.release()
    cv2.destroyAllWindows()
    end_time = time.time()
    print(f'Result: {frame_cnt} frames, {end_time-start_time} seconds, {frame_cnt/(end_time-start_time):.2f} fps')


def loop():
    if not capture.isOpened():
        logger.warning("카메라를 불러올 수 없습니다.")
        return
    ret, frame = capture.read()
    if not ret:
        logger.warning("영상을 불러올 수 없습니다.")
        return
    proc(frame)


def proc(frame: np.ndarray):
    frame = get_center_square(frame)
    cv2.imshow('RESULT', frame)

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if is_colored_pill(frame):
        gray_image = hsv_image[:,:,1]
    else:
        gray_image = hsv_image[:,:,2]

    bin_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[-1]
    bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, get_morph_kernel())
    bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, get_morph_kernel())

    pill_image = cv2.copyTo(frame, bin_image)
    pill_raw_color = extract_pill_color(pill_image)

    if pill_raw_color is None:
        logger.warning('감지된 색상이 없습니다.')
        return

    pill_color_name, pill_color = get_closest_colorname(pill_raw_color)

    contours = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    logger.info(f'{len(contours)} contours found.')
    if len(contours) > 1:
        logger.warning('아직 동시에 여러 개의 알약은 검출하기 어렵습니다.')
        return

    shape_name, shape_image = detect_shape(bin_image)

    out_frame = np.hstack([frame, shape_image], dtype=np.uint8)
    out_frame = np.vstack([out_frame, np.full((32, out_frame.shape[1], 3), pill_color[::-1], dtype=np.uint8)], dtype=np.uint8)
    out_frame = cv2.putText(out_frame, pill_color_name, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    out_frame = cv2.putText(out_frame, shape_name, (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('RESULT', out_frame)


def get_center_square(frame: np.ndarray) -> np.ndarray:
    center_y, center_x = frame.shape[0]/2, frame.shape[1]/2
    radius = 0.4 * min(frame.shape[:2])
    y_min = int(center_y-radius)
    y_max = int(center_y+radius)
    x_min = int(center_x-radius)
    x_max = int(center_x+radius)
    return cv2.resize(frame[y_min:y_max, x_min:x_max, :], (512, 512))


def is_colored_pill(frame: np.ndarray) -> bool:
    # TODO
    return True


@functools.cache
def get_morph_kernel() -> np.ndarray:
    return np.ones((9,9))


def extract_pill_color(bgr_frame: np.ndarray) -> typing.Optional[typing.Tuple[int]]:
    # colorgram.extract()는 이미지 크기에 비례한 수행시간을 필요로 하여
    # 이미지의 크기를 64x64로 다운샘플링하여 실시간 처리가 가능하도록 함.
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.resize(rgb_frame, (64, 64))
    pil_image = Image.fromarray(rgb_frame)
    colors = colorgram.extract(pil_image, number_of_colors=2)
    if len(colors) < 2:
        return None
    return colors[1].rgb


@functools.cache
def get_closest_colorname(color: webcolors.IntegerRGB) -> typing.Tuple[str, webcolors.IntegerRGB]:
    sample_colors = {
        'white': webcolors.hex_to_rgb('#ffffff'),
        'yellow': webcolors.hex_to_rgb('#ffeb3b'),
        'orange': webcolors.hex_to_rgb('#ff9800'),
        'pink': webcolors.hex_to_rgb('#ff65d5'),
        'red': webcolors.hex_to_rgb('#ba000d'),
        'brown': webcolors.hex_to_rgb('#964b00'),
        'lime': webcolors.hex_to_rgb('#7fe325'),
        'green': webcolors.hex_to_rgb('#006e1f'),
        'bluegreen': webcolors.hex_to_rgb('#0080a9'),
        'blue': webcolors.hex_to_rgb('#4269ff'),
        'navy': webcolors.hex_to_rgb('#1028ad'),
        'wine': webcolors.hex_to_rgb('#b90076'),
        'purple': webcolors.hex_to_rgb('#9b00b5'),
        # 'gray': webcolors.hex_to_rgb('#9e9e9e'),
        # 'black': webcolors.hex_to_rgb('#000000'),
    }
    def calc_dist(c1: webcolors.IntegerRGB, c2: webcolors.IntegerRGB) -> float:
        return math.sqrt(sum(map(lambda x: ((x[0]-x[1])**2), zip(c1, c2))))
    closest_colors = sorted(sample_colors.items(), key=lambda c: calc_dist(c[1], color))
    return closest_colors[0]


def detect_shape(bin_frame: np.ndarray) -> typing.Tuple[str, np.ndarray]:
    out_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    out_frame[bin_frame == 255, :] = 255

    contours = cv2.findContours(bin_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    logger.info(f'{len(contours)} contours found.')

    if len(contours) > 1:
        logger.warning('아직 동시에 여러 개의 알약은 검출하기 어렵습니다.')
        return 'not detected', out_frame

    contour = contours[0]
    approx = cv2.approxPolyDP(contour, CONTOUR_APPROX * cv2.arcLength(contour, True), True)
    n_approx = len(approx)  # 꼭짓점의 개수

    cv2.drawContours(out_frame, [contour], 0, (0, 0, 0), 2)
    for apr in approx:
        cv2.circle(out_frame, apr[0], radius=3, color=(0, 255, 0), thickness=2)

    if n_approx == 3:
        return 'triangle', out_frame
    elif n_approx == 4:
        return 'quadrilateral', out_frame
    elif n_approx == 5:
        return 'pentagon', out_frame
    elif n_approx < CIRCLE_MOMENTUM_THRESH:
        logger.info(f'{n_approx} points detected.')
        return 'oval? rectangular?', out_frame
    else:
        logger.info(f'{n_approx} points detected.')
        return 'circle??', out_frame


if __name__ == '__main__':
    setup()
