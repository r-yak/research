import collections

import numpy as np
import cv2

from .filters import inverse_gaussian_2d_filter
from .fft import fft_filter


STRUCTURING_ELEMENT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def by_saturation(image: np.ndarray) -> np.ndarray:
    gray_img = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]
    bin_img = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[-1]
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, STRUCTURING_ELEMENT)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, STRUCTURING_ELEMENT)
    return bin_img


def by_edge_detection(image: np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = fft_filter(gray_img, inverse_gaussian_2d_filter(gray_img.shape, 5))
    bin_img = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[-1]
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, STRUCTURING_ELEMENT)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, STRUCTURING_ELEMENT)
    bin_img = _remove_border_elements(bin_img)
    contours = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    out_img = np.zeros_like(bin_img)
    cv2.drawContours(out_img, contours, -1, 255, -1)
    return out_img


def _remove_border_elements(bin_image: np.ndarray) -> np.ndarray:
    q = collections.deque()
    h, w = bin_image.shape[:2]
    for x in range(w):
        q.append((0, x))
        q.append((h-1, x))
    for y in range(h):
        q.append((y, 0))
        q.append((y, w-1))
    while q:
        y, x = q.popleft()
        _remove_border_elements_util(bin_image, y, x)
    return bin_image


def _remove_border_elements_util(mat: np.ndarray, y: int, x: int):
    if not _check_indicies(mat, y, x):
        return
    if not mat[y, x]:
        return
    mat[y, x] = 0
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        _remove_border_elements_util(mat, y+dy, x+dx)


def _check_indicies(mat: np.ndarray, *args: int) -> bool:
    if len(args) > len(mat.shape):
        raise ValueError()
    for arg in args:
        if arg < 0:
            return False
    for axis, arg in enumerate(args):
        if arg >= mat.shape[axis]:
            return False
    return True