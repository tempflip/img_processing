from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import numpy as np
import matplotlib.pyplot as plt


img = np.zeros((100, 150), dtype=bool)
img[30, :] = 1
img[:, 65] = 1
img[35:45, 35:50] = 1


for i in range(90):
	img[i, i] = 1
img += np.random.random(img.shape) > 0.95


out, angles, d = hough_line(img)
print d

plt.imshow(out)
plt.show()