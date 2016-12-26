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


		if b > vmax or b < vmin : continue
		b+= vmax-1


		feature_space[i, b] += 1

	return feature_space

def space_point_to_line(p, feature_space, M_MAX = 20, B_COEFF = 0.1):
	space = np.linspace(- M_MAX, M_MAX, len(feature_space))
	m = space[p[0]]
	b = p[1] - len(feature_space[0]) / 2

	return m, b

def get_coordinates(feature_space, val):
	points = []
	coords = np.where(feature_space == val)

	for i in range(len(coords[0])):

		points.append((coords[0][i], coords[1][i]))
	return points

def build_line(shape, m, b, val = 99):
	im = np.zeros(shape)
	for x in range(shape[1]):
		point_y = int(m * x + b)
		if point_y < 0 or point_y >= shape[0] : continue

		im[point_y][x] = val
	return im


def run():
	M_MAX = 2
	B_COEFF = 1

	im = np.zeros((100, 100))
	feature_space = np.zeros((200, 200)) # d, theta

	points = [(10, 10), (20, 20), (30,30), (40, 40), (50, 50), (40, 32), (12, 76), (5, 5), (50, 10), (55, 15), (60, 20), (70, 30)]

	points1 = [(10, 10), (20, 20), (30,30), (40, 40)]
	points2 = [(10, 10), (15, 20), (20,30), (25, 40)]
	points3 = [(50, 10), (60, 20), (70,30), (80, 40)]

	points4 = [(1,1), (10, 50)]



	#points = [(10, 40), (50, 77), (33, 80), (40, 75), (90, 70)]

	for p in points[:]:
		im[p[0], p[1]] = 1
		feature_space = add_to_fetaure_space_linear(p, feature_space, M_MAX = M_MAX, B_COEFF = B_COEFF)


	#best_matches = feature_space == feature_space.max()

	val_set = list(set(feature_space.flatten()))

	candidates = []

	candidates += get_coordinates(feature_space, val_set[-1])
	candidates += get_coordinates(feature_space, val_set[-2])




	proposal = np.array(im)

	for p in candidates:
		print "candidate:", p
		m, b = space_point_to_line(p, feature_space, M_MAX = M_MAX, B_COEFF = B_COEFF)

		line = build_line(proposal.shape, m, b, val = 3)
		proposal += line




	plt.subplot(1,3,1)
	plt.imshow(im, interpolation="none")
	plt.subplot(1,3,2)
	plt.imshow(feature_space.T, interpolation="none")
	plt.subplot(1,3,3)
	plt.imshow(proposal, interpolation="none")


	plt.show()


if __name__ == "__main__":
	run()
