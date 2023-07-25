import math
import typing
import colorgram
import cv2
import numpy as np
import matplotlib.pyplot as plt
import webcolors
from PIL import Image


image = cv2.imread('../images/IMG_9653.png')


center_y, center_x = image.shape[0]/2, image.shape[1]/2
radius = 0.4 * min(image.shape[:2])

y_min = int(center_y-radius)
y_max = int(center_y+radius)
x_min = int(center_x-radius)
x_max = int(center_x+radius)

image = image[y_min:y_max, x_min:x_max, :]
image = cv2.resize(image, (512, 512))


rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(rgb_image)

fig = plt.figure()

ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(rgb_image)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(r, g, b, marker='.')
ax2.set_xlabel('R')
ax2.set_ylabel('G')
ax2.set_zlabel('B')
ax2.view_init(45, -135)

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(r, g, b, marker='.')
ax3.set_xlabel('R')
ax3.set_ylabel('G')
ax3.set_zlabel('B')
ax3.view_init(30, -45)


hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)


gray_image = s

plt.imshow(gray_image, cmap='gray')


plt.hist(gray_image)


gray_image_equalized = cv2.equalizeHist(gray_image)

plt.subplot(121)
plt.hist(gray_image_equalized)
plt.subplot(122)
plt.imshow(gray_image_equalized, cmap='gray')


otsu_value, bin_image = cv2.threshold(
    gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Otsu's threshold value: {otsu_value}")
plt.imshow(bin_image, cmap='gray')


kernel = np.ones((9, 9))
bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel)

plt.imshow(bin_image, cmap='gray')


pill_image = cv2.copyTo(rgb_image, bin_image)

down_sampled_image = cv2.resize(pill_image, (64, 64))

pillow_image = Image.fromarray(down_sampled_image)
colors: typing.List[colorgram.Color]
colors = colorgram.extract(pillow_image, number_of_colors=2)

pill_color = colors[1].rgb

plt.imshow(np.full((1, 1, 3), pill_color))


sample_colors = {
    'white': webcolors.hex_to_rgb('#ffffff'),
    'yellow': webcolors.hex_to_rgb('#ffeb3b'),
    'orange': webcolors.hex_to_rgb('#ff9800'),
    'pink': webcolors.hex_to_rgb('#ff65d5'),
    'red': webcolors.hex_to_rgb('#ff0000'),
    'brown': webcolors.hex_to_rgb('#ab4723'),
    'lime': webcolors.hex_to_rgb('#8bc34a'),
    'green': webcolors.hex_to_rgb('#00962f'),
    'bluegreen': webcolors.hex_to_rgb('#0080a9'),
    'blue': webcolors.hex_to_rgb('#4269ff'),
    'navy': webcolors.hex_to_rgb('#1028ad'),
    'wine': webcolors.hex_to_rgb('#b90076'),
    'purple': webcolors.hex_to_rgb('#9b00b5'),
    'gray': webcolors.hex_to_rgb('#9e9e9e'),
    'black': webcolors.hex_to_rgb('#000000'),
}


def calc_color_distance(c1, c2):
    return math.sqrt(sum(map(lambda x: ((x[0]-x[1])**2), zip(c1, c2))))


closest_colors = sorted(sample_colors.items(
), key=lambda color: calc_color_distance(color[1], pill_color))
print(closest_colors)

pill_color = closest_colors[0][1]
pill_color_name = closest_colors[0][0]

plt.imshow(np.full((1, 2, 3), pill_color))
plt.title(pill_color_name)


contours = cv2.findContours(
    bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]


print(f'{len(contours)} contours found.')

if len(contours) > 1:
    raise Exception('아직 동시에 여러 개의 알약은 검출하기 어렵습니다.')

contour = contours[0]
approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
n_approx = len(approx)  # 꼭짓점의 개수

if n_approx == 3:
    print('triangle')
elif n_approx == 4:
    print('quadrilateral')
elif n_approx == 5:
    print('pentagon')
else:
    print('circle? oval? rectangular?')
    print(n_approx)


temp_image = np.zeros((512, 512, 3), dtype=np.uint8)
temp_image[bin_image == 255, :] = 255
cv2.drawContours(temp_image, [contour], 0, (255, 0, 0), 2)

x = approx[:, :, 0]
y = approx[:, :, 1]

plt.subplot(121)
plt.imshow(bin_image, cmap='gray')

plt.subplot(122)
plt.imshow(temp_image, cmap='gray')
plt.scatter(x, y, marker='o')


def create_pill_mask(image, is_colored=True):
    pass
