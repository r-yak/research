import typing

import cv2
import colorgram
import numpy

from PIL import Image


KEY_CODE_ESC = 27

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
    ret, frame = capture.read()
    if not ret:
        return
    cv2.imshow('CAMERA', frame)
    show_colors(extract_color(frame))


def extract_color(frame: numpy.ndarray) -> typing.List[colorgram.Color]:
    number_of_colors = 6
    down_sample_dst_size = (64, 64)
    down_sampled_frame = cv2.resize(frame, down_sample_dst_size)
    pil_image = Image.fromarray(down_sampled_frame)
    return colorgram.extract(pil_image, number_of_colors)


def show_colors(colors: typing.List[colorgram.Color]) -> None:
    block_size = 64
    frame = numpy.zeros(
        (block_size, block_size * len(colors), 3), dtype=numpy.uint8)
    for i, color in enumerate(colors):
        frame[:, block_size*i:block_size*(i+1)] = color.rgb
    cv2.imshow('COLOR', frame)


if __name__ == '__main__':
    setup()
