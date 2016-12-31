from scipy import misc, signal, stats
import matplotlib.pyplot as plt 
import numpy as np
from math import *
from hough_transform import *

img = np.array(misc.imresize(misc.imread('edge1.png', mode = 'L'), (100, 100)), dtype="float32")
img = img[5:-5, 5:-5]

x = np.zeros(img.shape)
x[img > 20] += 1
img = x



#img = np.zeros((100, 100))
#img[10,50] = 1
#img[50, 70] = 1

hough_space = get_hough_space_from_image(img, val=1)

val_set = list(set(hough_space.flatten()))
highest_vals = val_set[-5:]

img_lines = np.zeros(img.shape)
for val in highest_vals:
	for coord in get_coordinates(hough_space, val):
		theta, d = coord_to_polar(coord, hough_space)
		print '->', coord, theta, d
		img_lines += build_line_polar(theta, d, shape = img_lines.shape, val=5)
		#img_lines += build_line_polar(1.1, 60, shape = img_lines.shape)
		#img_lines += build_line_polar(1, 70, shape = img_lines.shape)

img_lines += img

plt.subplot(1,3,1)
plt.imshow(img)

plt.subplot(1,3,2)
plt.imshow(hough_space)

plt.subplot(1,3,3)
plt.imshow(img_lines)

plt.show()

