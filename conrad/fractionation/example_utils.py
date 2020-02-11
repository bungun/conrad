import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from data_utils import circle, ellipse

def simple_structures(m_grid, n_grid):
	# Create polar grid.
	theta = np.linspace(0, 2*np.pi, m_grid)
	r = np.linspace(0, 1, n_grid)
	theta_grid, r_grid = np.meshgrid(theta, r)
	x_grid = r_grid*np.cos(theta_grid)
	y_grid = r_grid*np.sin(theta_grid)

	# Define structure regions.
	# Body voxels (k = 3).
	regions = np.full((n_grid, m_grid), 3)

	# PTV (k = 0).
	r0 = (0.5 + 0.65)/2
	theta0_l = 7*np.pi/16
	theta0_r = np.pi/16
	r_width = (0.65 - 0.5)/2
	circle_l = circle(x_grid, y_grid, (r0*np.cos(theta0_l), r0*np.sin(theta0_l)), r_width) <= 0
	circle_r = circle(x_grid, y_grid, (r0*np.cos(theta0_r), r0*np.sin(theta0_r)), r_width) <= 0
	slice_c = (r_grid >= 0.5) & (r_grid <= 0.65) & (theta_grid <= theta0_l) & (theta_grid >= theta0_r)
	regions[circle_l | circle_r | slice_c] = 0
	# regions[(r_grid >= 0.5) & (r_grid <= 0.65) & (theta_grid <= np.pi/2)] = 0

	# OAR (k = 1).
	# regions[circle(x_grid, y_grid, (0, 0), 0.35) <= 0] = 1
	regions[r_grid <= 0.35] = 1

	# OAR (k = 2).
	r0 = 0.6
	theta0 = 7*np.pi/6
	x0 = r0*np.cos(theta0)
	y0 = r0*np.sin(theta0)
	x_width = 0.075
	y_width = 0.15
	regions[ellipse(x_grid, y_grid, (x0, y0), (x_width, y_width), np.pi/6) <= 0] = 2
	return theta_grid, r_grid, regions

def simple_colormap():
	K = 4
	struct_cmap = ListedColormap(['red', 'blue', 'green', 'white'])
	struct_bounds = np.arange(K+1) - 0.5
	# struct_cmap = truncate_cmap(plt.cm.rainbow, 0, 0.1*K, n = K)
	# struct_bounds = np.linspace(0, K, K+1)
	struct_norm = BoundaryNorm(struct_bounds, struct_cmap.N)
	struct_kw = {"cmap": struct_cmap, "norm": struct_norm, "alpha": 0.5}
	return struct_kw
