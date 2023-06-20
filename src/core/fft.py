import numpy as np
import cv2


def fft_filter(gray: np.ndarray, filter: np.ndarray) -> np.ndarray:
    if len(gray.shape) != 2:
        raise ValueError()
    fft = np.fft.fft2(np.float32(gray))
    fft_shift = np.fft.fftshift(fft)
    fft_ishift = np.fft.ifftshift(fft_shift * filter)
    ifft = np.fft.ifft2(fft_ishift)
    return np.uint8(cv2.magnitude(ifft.real, ifft.imag))