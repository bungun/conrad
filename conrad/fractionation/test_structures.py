import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from example_utils import *
from data_utils import line_integral_mat, line_segments
from plot_utils import plot_structures, plot_beams, transp_cmap

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

xlim = (-1,1)
ylim = (-1,1)

x_grid, y_grid, structures = simple_structures(m_grid, n_grid, xlim, ylim)

if SHOW_STRUCTS:
	struct_kw = simple_colormap()
	plot_structures(x_grid, y_grid, structures, **struct_kw)

if CALC_AMAT:
	A, angles, d_vec = line_integral_mat(structures, angles = n_angle, n_bundle = n_bundle, offset = offset)
	if PLOT_LINES:
		beams = np.full((1, n_angle*n_bundle), 1)
		struct_kw = simple_colormap()
		struct_tuple = (x_grid, y_grid, structures)
		plot_beams(beams, angles, d_vec, xlim = xlim, ylim = ylim)
		# plot_beams(beams, angles, d_vec, structures = struct_tuple, struct_kw = struct_kw, xlim = xlim, ylim = ylim)
	if SAVE_AMAT:
		np.save("data/A_cardiod_rot_{0}x{1}-grid_{2}-angles_{3}-bundles.npy".format(m_grid, n_grid, n_angle, n_bundle), A)
