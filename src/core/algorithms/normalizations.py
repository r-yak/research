from __future__ import annotations

import abc
import functools
import typing

import numpy as np
import cv2


def normalize(image: np.ndarray) -> np.ndarray:
    for normalizer in get_normalizers():
        image = normalizer.normalize(image)
    return image


@functools.cache
def get_normalizers() -> typing.List[Normalizer]:
    return [
        ShapeNormalizer(),
        ColorNormalizer(),
    ]


class Normalizer(abc.ABC):
    @abc.abstractmethod
    def normalize(self, image: np.ndarray) -> np.ndarray:
        ...


class ShapeNormalizer(Normalizer):
    SQUARED_FRAME_SIZE = 512
    SQUARED_FRAME_RADIUS_RATIO = 0.4

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return self.square_crop(image)

    def square_crop(self, image: np.ndarray) -> np.ndarray:
        cy, cx = image.shape[0]/2, image.shape[1]/2
        radius = min(image.shape[:2])
        radius *= ShapeNormalizer.SQUARED_FRAME_RADIUS_RATIO
        xmin = int(cx-radius)
        xmax = int(cx+radius)
        ymin = int(cy-radius)
        ymax = int(cy+radius)
        if len(image.shape) > 2:
            return image[ymin:ymax, xmin:xmax, :]
        return image[ymin:ymax, xmin:xmax]


class ColorNormalizer(Normalizer):
    def normalize(self, image: np.ndarray) -> np.ndarray:
        return self.applyGrayworldWB(image)

    def applyGrayworldWB(self, image: np.ndarray) -> np.ndarray:
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image_balanced = self.createGrayworldWB().balanceWhite(lab_image)
        return cv2.cvtColor(image_balanced, cv2.COLOR_LAB2BGR)

    @functools.cache
    def createGrayworldWB(self):
        return cv2.xphoto.createGrayworldWB()
