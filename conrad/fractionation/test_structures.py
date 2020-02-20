import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from example_utils import *
from data_utils import line_integral_mat, line_segments
from plot_utils import plot_structures

SHOW_STRUCTS = False
CALC_AMAT = True
SAVE_AMAT = False
PLOT_LINES = True

# m_grid = 10000
m_grid = 1000
n_grid = 1000

# n = 1000
n_angle = 10
n_bundle = 5
offset = 0.01

x_grid, y_grid, structures = simple_structures(m_grid, n_grid)

if SHOW_STRUCTS:
	struct_kw = simple_colormap()
	plot_structures(x_grid, y_grid, regions, **struct_kw)
	

if CALC_AMAT:
	A, angles, d_vec = line_integral_mat(structures, angles = n_angle, n_bundle = n_bundle, offset = offset)
	if PLOT_LINES:
		fig, ax = plt.subplots()
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		
		segments = line_segments(angles, d_vec, xlim = (-1,1), ylim = (-1,1))
		lc = LineCollection(segments)
		ax.add_collection(lc)
		plt.show()
	if SAVE_AMAT:
		np.save("data/A_cardiod_rot_{0}x{1}-grid_{2}-angles_{3}-bundles.npy".format(m_grid, n_grid, n_angle, n_bundle), A)
