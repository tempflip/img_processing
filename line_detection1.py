from scipy import misc, signal, stats
import matplotlib.pyplot as plt 
import numpy as np
from math import *
from hough_transform import *

img = np.array(misc.imresize(misc.imread('edge1.png', mode = 'L'), (100, 100)), dtype="float32")

img = np.array(img[60:, :])
im = np.array(img)

M_MAX = 2
B_COEFF = 1

feature_space = np.zeros((200, 200)) 


for i, row in enumerate(im):
	for j, pix in enumerate(row):
		if pix < 20  : 
			im[i,j] = 0
		else : 
			im[i,j] = 1
			feature_space = add_to_fetaure_space_linear((i, j), feature_space, M_MAX = M_MAX, B_COEFF = B_COEFF)



val_set = list(set(feature_space.flatten()))
print val_set
candidates = []

CANDIDATE_NUMBER = min(20, len(val_set))
SLOPE_THRESHOLD = 0.3

i = 0
while i < CANDIDATE_NUMBER: 
	candidates = get_coordinates(feature_space, val_set[-i - 1])
	i += 1

proposal = np.zeros(im.shape)
#proposal = np.array(img)


for p in candidates:

	m, b = space_point_to_line(p, feature_space, M_MAX = M_MAX, B_COEFF = B_COEFF)

	if abs(m) < SLOPE_THRESHOLD : continue
	#print "candidate:", m, b


	line = build_line(proposal.shape, m, b, val = 10)


	proposal += line


plt.subplot(2,1,1)
plt.imshow(img, interpolation="none")

#plt.subplot(1,3,2)
#plt.imshow(feature_space, interpolation="none")

plt.subplot(2,1,2)
plt.imshow(proposal, interpolation="none")

plt.show()