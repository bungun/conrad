import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from data_utils import circle, ellipse, cardioid, limacon

def simple_structures(m_grid, n_grid, xlim = (-1,1), ylim = (-1,1)):
	# Create polar grid.
	x = np.linspace(xlim[0], xlim[1], m_grid)
	y = np.linspace(ylim[0], ylim[1], n_grid)
	x_grid, y_grid = np.meshgrid(x, y)
	r_grid = np.sqrt(x_grid**2 + y_grid**2)
	theta_grid = np.arctan2(y_grid, x_grid)

	# Define structure regions.
	# Body voxels (s = 4).
	regions = np.full((n_grid, m_grid), 4)

	# PTV (s = 0).
	# regions[circle(x_grid, y_grid, (0,0), 0.35) <= 0] = 0
	regions[cardioid(x_grid, y_grid, 0.125, (-0.1,0), np.pi) <= 0] = 0

	# OAR (s = 1).
	r_inner = 0.7
	r_outer = 0.85
	r0 = (r_inner + r_outer)/2
	theta0_l = 3*np.pi/8
	theta0_r = np.pi/8
	r_width = (r_outer - r_inner)/2
	circle_l = circle(x_grid, y_grid, (r0*np.cos(theta0_l), r0*np.sin(theta0_l)), r_width) <= 0
	circle_r = circle(x_grid, y_grid, (r0*np.cos(theta0_r), r0*np.sin(theta0_r)), r_width) <= 0
	slice_c = (r_grid >= r_inner) & (r_grid <= r_outer) & (theta_grid <= theta0_l) & (theta_grid >= theta0_r)
	regions[circle_l | circle_r | slice_c] = 1
	# regions[(r_grid >= 0.5) & (r_grid <= 0.65) & (theta_grid <= np.pi/2)] = 1

	# OAR (s = 2).
	x0 = -0.275  # -0.375
	y0 = 0.45  # 0.65
	x_width = 0.15  # 0.1
	y_width = 0.25  # 0.18
	regions[ellipse(x_grid, y_grid, (x0, y0), (x_width, y_width), np.pi/3) <= 0] = 2
	
	# OAR (s = 3).
	x0 = -0.1
	y0 = -0.7
	x_width = 0.2
	y_width = 0.15
	regions[ellipse(x_grid, y_grid, (x0, y0), (x_width, y_width)) <= 0] = 3
	return x_grid, y_grid, regions

def simple_colormap(one_idx = False):
	K = 5
	struct_cmap = ListedColormap(['red', 'blue', 'green', 'orange', 'white'])
	struct_bounds = np.arange(K+1) - 0.5 + int(one_idx)
	# struct_cmap = truncate_cmap(plt.cm.rainbow, 0, 0.1*K, n = K)
	# struct_bounds = np.linspace(0, K, K+1)
	struct_norm = BoundaryNorm(struct_bounds, struct_cmap.N)
	struct_kw = {"cmap": struct_cmap, "norm": struct_norm, "alpha": 0.75}
	return struct_kw
