import math
import logging
import typing

import webcolors
import numpy as np
import colorgram
import cv2
from PIL import Image

from .enums import Color, Shape


class ColorNotDetectedException(Exception):
    pass


class ShapeNotDetectedException(Exception):
    pass


class ColorDetector:
    def __init__(self) -> None:
        self.logger = logging.Logger('color detector')

    def detect(self, image: np.ndarray) -> Color:
        return self._find_closest_color(image)

    def get_actual_color(self) -> webcolors.IntegerRGB:
        return self.actual_color

    def _find_closest_color(self, bgr_image: np.ndarray) -> Color:
        try:
            self.actual_color = self._find_color(bgr_image)
        except ColorNotDetectedException:
            return Color.NONE
        return self._find_by_rgb_distance()

    def _find_color(self, bgr_image: np.ndarray) -> webcolors.IntegerRGB:
        rgb_frame = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.resize(rgb_frame, (64, 64))
        pil_image = Image.fromarray(rgb_frame)
        colors = colorgram.extract(pil_image, number_of_colors=2)
        if len(colors) < 2:
            self.logger.warning('감지된 색상이 없습니다.')
            raise ColorNotDetectedException()
        return colors[1].rgb

    def _find_by_uv_distance(self):
        return sorted(Color, key=lambda c: self._calc_uv_dist(c.value, self.actual_color))[0]

    def _calc_uv_dist(self, c1: webcolors.IntegerRGB, c2: webcolors.IntegerRGB) -> float:
        yuv = cv2.cvtColor(np.array([[c1, c2]], dtype=np.uint8), cv2.COLOR_RGB2YUV)
        c1, c2 = yuv[0,:,1:]
        return self._calc_nd_dist(c1, c2)

    def _find_by_rgb_distance(self):
        return sorted(Color, key=lambda c: self._calc_nd_dist(c.value, self.actual_color))[0]

    def _calc_nd_dist(self, v1: typing.Iterable[int], v2: typing.Iterable[int]) -> float:
        return math.sqrt(sum(map(lambda x: ((x[0]-x[1])**2), zip(v1, v2))))


class ShapeDetector:
    def __init__(self) -> None:
        self.APPROX_EPSILON = 0.04
        self.CIRCLE_MOMENTUM_THRESH = 7
        self.logger = logging.Logger('shape detector')

    def detect(self, image: np.ndarray) -> Shape:
        self.update_contour(image)
        try:
            return self._determine_shape()
        except ShapeNotDetectedException:
            return Shape.NONE

    def update_contour(self, image: np.ndarray):
        self.contours = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if self.contours:
            self.approx = cv2.approxPolyDP(
                self.contours[0], self.APPROX_EPSILON * cv2.arcLength(self.contours[0], True), True)
        else:
            self.approx = []

    def get_contour(self):
        return self.contours[0]

    def get_vertices(self):
        return self.approx

    def _determine_shape(self) -> Shape:
        if len(self.contours) > 1:
            self.logger.warning('아직 동시에 여러 개의 알약은 검출하기 어렵습니다.')
            raise ShapeNotDetectedException()
        n_approx = len(self.approx)
        if n_approx == 3:
            return Shape.TRIANGLE
        elif n_approx == 4:
            return Shape.QUADRILATERAL
        elif n_approx == 5:
            return Shape.PENTAGON
        elif n_approx < self.CIRCLE_MOMENTUM_THRESH:
            self.logger.info(f'{n_approx} points detected.')
            return Shape.OVAL_LIKE
        else:
            self.logger.info(f'{n_approx} points detected.')
            return Shape.CIRCLE


colorDetector = ColorDetector()
shapeDetector = ShapeDetector()
