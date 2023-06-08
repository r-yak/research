import typing

import cv2
import numpy as np
import colorgram

from PIL import Image


def extract_color(frame: np.ndarray) -> typing.List[colorgram.Color]:
    number_of_colors = 6
    down_sample_dst_size = (64, 64)
    down_sampled_frame = cv2.resize(frame, down_sample_dst_size)
    pil_image = Image.fromarray(down_sampled_frame)
    return colorgram.extract(pil_image, number_of_colors)


def make_pallete(colors: typing.List[colorgram.Color]) -> np.ndarray:
    size = 64
    frame = np.zeros((size, size * len(colors), 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        frame[:, size*i:size*(i+1)] = color.rgb
    return frame
