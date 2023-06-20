import functools
import typing

import numpy as np


@functools.cache
def inverse_gaussian_2d_filter(shape: typing.Tuple[int], radius: int) -> np.ndarray:
    return 1 - gaussian_2d_filter(shape, radius)


def gaussian_2d_filter(shape: typing.Tuple[int], radius: int) -> np.ndarray:
    h, w = shape[:2]
    cy, cx = h/2, w/2
    X, Y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    return np.exp(-(np.power(X-cx, 2)+np.power(Y-cy, 2)) / (2*np.power(radius, 2)))
