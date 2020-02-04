import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from data_utils import line_integral_mat

n = 10   # Number of beams.
m_grid = 10000
n_grid = 500

r0 = 0.6
theta0 = 7*np.pi/6
x0 = r0*np.cos(theta0)
y0 = r0*np.sin(theta0)
x_width = 0.075
y_width = 0.15

def ellipse(x, y, center = (0,0), width = (1,1), angle = 0):
	x0, y0 = center
	x_width, y_width = width
	return (((x - x0)*np.cos(angle) + (y - y0)*np.sin(angle))/x_width)**2 + \
		   (((x - x0)*np.sin(angle) - (y - y0)*np.cos(angle))/y_width)**2 - 1

# Create polar grid.
theta = np.linspace(0, 2*np.pi, m_grid)
r = np.linspace(0, 1, n_grid)
theta_grid, r_grid = np.meshgrid(theta, r)

# Define structure regions.
regions = np.full((n_grid, m_grid), 3)   # Body voxels (k = 3).

# PTV (k = 0).
regions[(r_grid >= 0.5) & (r_grid <= 0.65) & (theta_grid <= np.pi/2)] = 0

# OAR (k = 1).
regions[r_grid <= 0.35] = 1

# OAR (k = 2).
x_grid = r_grid*np.cos(theta_grid)
y_grid = r_grid*np.sin(theta_grid)
regions[ellipse(x_grid, y_grid, (x0, y0), (x_width, y_width), np.pi/6) <= 0] = 2
K = np.unique(regions).size

# Create colormap for regions.
cmap = ListedColormap(['red', 'blue', 'green', 'white'])
bounds = np.arange(K+1) - 0.5
norm = BoundaryNorm(bounds, cmap.N)

# Plot regions.
fig, ax = plt.subplots(1, 1, subplot_kw = dict(projection = "polar"))
ctf = ax.contourf(theta_grid, r_grid, regions, cmap = cmap, norm = norm)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_theta_zero_location("N")
plt.colorbar(ctf)
# plt.show()

# Plot beams.
# beam_angles = np.linspace(0, np.pi, n+1)[:-1]
# arr = np.ones((2*n, 2))
# arr[0::2,0] = beam_angles
# arr[1::2,0] = beam_angles + np.pi
# segments = np.split(arr, n)
# lc = LineCollection(segments, color = "grey")
# ax.add_collection(lc)
# plt.show()

# Define problem.
m = 1000
n = 100
T_treat = 20
T_recov = 14
T = T_treat + T_recov

A = line_integral_mat(theta_grid, regions, beam_angles = n, atol = 1e-4)
A_list = T_treat*[A]

F = np.diag([1.02, 0.95, 0.90, 0.75])
G = -np.eye(K)
r = np.zeros(K)
h_init = np.array([1] + (K-1)*[0])
