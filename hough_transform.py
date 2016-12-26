from scipy import misc, signal, stats
import matplotlib.pyplot as plt 
import numpy as np
from math import *

def add_to_fetaure_space_linear(p, feature_space, M_MAX = 20, B_COEFF = 0.1):
	#### drawing the y = mx + b point represantation in the fateure space

	vmin = - len(feature_space[0]) / 2
	vmax = len(feature_space[0]) / 2

	space = np.linspace(- M_MAX, M_MAX, len(feature_space))

	for i in range(len(space)):
		m = space[i]
		y = p[0]
		x = p[1]

		b = int((y - m * x) * B_COEFF) 

		#print "i", i, "m", m, "b", b

		if b > vmax or b < vmin : continue
		b+= vmax-1

		print "############i", i, "m", m, "b", b

		feature_space[i, b] += 1

	return feature_space

def space_point_to_line(p, M_MAX = 20, B_COEFF = 0.1):
	pass

def get_coordinates(feature_space, val):
	points = []
	coords = np.where(feature_space == val)

	for i in range(len(coords[0])):

		points.append((coords[0][i], coords[1][i]))
	return points

"""
def add_to_fetaure_space(p, feature_space):

	N = 10

	d = sqrt(p[0] ** 2 + p[1] ** 2)
	#print "d", d

	for i in range(len(feature_space)-1):
		d = i+1 + 0.0
		sin_p = p[0] / d
		cos_p = p[1] / d

		if sin_p > 1 or cos_p > 1 : continue 

		#print "## theta feature at d = {}, sin_p = {}, cost_p = {}".format(d, sin_p, cos_p)
		theta = asin(sin_p)

		feature_value = int(theta * 40)
		#print "####### theta value : {} at point d {}".format(theta, d)

		feature_space[d / N, feature_value] = feature_space[d / N, feature_value] + 1
	return feature_space
"""


M_MAX = 20
B_COEFF = 0.1

im = np.zeros((100, 100))
feature_space = np.zeros((200, 200)) # d, theta

points = [(10, 10), (20, 20), (30,30), (40, 40), (50, 50), (40, 32), (12, 76), (5, 5), (50, 10), (55, 15), (60, 20), (70, 30)]

#points = [(10, 40), (50, 77), (33, 80), (40, 75), (90, 70)]

for p in points[:]:
	im[p[0], p[1]] = 1
	feature_space = add_to_fetaure_space_linear(p, feature_space, M_MAX = M_MAX, B_COEFF = B_COEFF)


#best_matches = feature_space == feature_space.max()

val_set = list(set(feature_space.flatten()))

print val_set

candidates = get_coordinates(feature_space, val_set[-1])
for p in candidates:
	print feature_space[p[0], p[1]]

	print space_point_to_line(p, M_MAX = M_MAX, B_COEFF = B_COEFF)

#exit()



plt.subplot(1,2,1)
plt.imshow(im, interpolation="none")
plt.subplot(1,2,2)
plt.imshow(feature_space.T, interpolation="none")
#plt.subplot(3,1,3)
#plt.imshow(best_matches.T, interpolation="none")


plt.show()

