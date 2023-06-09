import logging
import cv2
import time
from core.color import *


KEY_CODE_ESC = 27

logger = logging.Logger('app')
capture: cv2.VideoCapture


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
    # 채도에 대한 채널만 추출
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray_image = cv2.split(hsv_image)[1]

    # 빠른 처리속도를 위해 다운샘플링
    gray_image = cv2.resize(gray_image, dsize=(256, 256))

    # 블러링을 통한 노이즈 감소
    gray_image = cv2.GaussianBlur(gray_image, ksize=(5,5), sigmaX=1)

    # 임계값 처리 (흰 배경 = 낮은 채도이므로 가능한 알고리즘)
    bin_frame = cv2.threshold(gray_image, thresh=40, maxval=255, type=cv2.THRESH_BINARY)[1]

    # 모폴로지 침식 연산을 통한 노이즈 감소
    bin_frame = cv2.erode(bin_frame, kernel=(5,5), iterations=2)

    # 마스크 이미지 생성 (윤곽선 검출 -> 내부 공간 채우기)
    mask_image = np.zeros_like(bin_frame)
    contours = cv2.findContours(bin_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(mask_image, contours, contourIdx=-1, color=255, thickness=-1)

    # 마스킹 처리를 위해 업샘플링으로 크기 복구
    mask_image = cv2.resize(mask_image, dsize=frame.shape[1::-1])

    # 마스킹 처리 (유색 알약만 추출 됨)
    colored_pill_image = cv2.copyTo(frame, mask_image)

    # 알약의 색 검출
    colors = extract_colors(colored_pill_image, 2)

    if len(colors) < 2:
        logger.warning('알약이 검출되지 않았습니다.')
        primary_color = colorgram.Color(0, 0, 0, 1.0) # black
    else:
        logger.info('알약이 검출되었습니다.')
        background_color = colors[0] # in the most cases, black.
        primary_color = colors[1]

    # 결과 출력
    row1 = np.hstack([frame, colored_pill_image])
    row2 = np.zeros_like(row1)
    row2[:,:] = primary_color.rgb[::-1] # RGB->BGR

    cv2.imshow('RESULT', np.vstack([row1, row2]))


def extract_colors(bgr_frame: np.ndarray, number_of_colors:int=6) -> typing.List[colorgram.Color]:
    # colorgram.extract()는 이미지 크기에 비례한 수행시간을 필요로 하여
    # 이미지의 크기를 64x64로 다운샘플링하여 실시간 처리가 가능하도록 함.
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    down_sample_dst_size = (64, 64)
    down_sampled_frame = cv2.resize(rgb_frame, down_sample_dst_size)
    pil_image = Image.fromarray(down_sampled_frame)
    return colorgram.extract(pil_image, number_of_colors)


def make_pallete(colors: typing.List[colorgram.Color]) -> np.ndarray:
    size = 64
    frame = np.zeros((size, size * len(colors), 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        frame[:, size*i:size*(i+1)] = color.rgb
    return frame


def freq_2d_filter(gray: np.ndarray, filter: np.ndarray) -> np.ndarray:
    if len(gray.shape) != 2:
        raise ValueError()
    fft = np.fft.fft2(np.float32(gray))
    fft_shift = np.fft.fftshift(fft)
    fft_ishift = np.fft.ifftshift(fft_shift * filter)
    ifft = np.fft.ifft2(fft_ishift)
    return np.uint8(cv2.magnitude(ifft.real, ifft.imag))


def gaussian_2d_filter(shape: typing.Tuple[int], radius: int) -> np.ndarray:
    h, w = shape[:2]
    cy, cx = h/2, w/2
    X, Y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    out = np.exp(-(np.power(X-cx,2)+np.power(Y-cy,2))/(2*np.power(radius,2)))
    return out


if __name__ == '__main__':
    setup()
