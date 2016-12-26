from scipy import misc, signal, stats
import matplotlib.pyplot as plt 
import numpy as np
from math import *
from hough_transform import *

im = np.array(misc.imresize(misc.imread('lines1.png', mode = 'L'), (100, 100)), dtype="float32")


M_MAX = 2
B_COEFF = 1

feature_space = np.zeros((200, 200)) 

for i, row in enumerate(im):
	for j, pix in enumerate(row):
		if pix == 255  : 
			im[i,j] = 0
		else : 
			im[i,j] = 1
			feature_space = add_to_fetaure_space_linear((i, j), feature_space, M_MAX = M_MAX, B_COEFF = B_COEFF)



val_set = list(set(feature_space.flatten()))
print val_set
candidates = []


for i in range(30):
	candidates += get_coordinates(feature_space, val_set[-i - 1])



proposal = np.array(im)

for p in candidates:
	print "candidate:", p
	m, b = space_point_to_line(p, feature_space, M_MAX = M_MAX, B_COEFF = B_COEFF)
	line = build_line(proposal.shape, m, b, val = 3)
	proposal += line


plt.subplot(1,3,1)
plt.imshow(im, interpolation="none")

plt.subplot(1,3,2)
plt.imshow(feature_space, interpolation="none")

plt.subplot(1,3,3)
plt.imshow(proposal, interpolation="none")

plt.show()