import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from data_utils import line_integral_mat

n = 10   # Number of beams.
m_grid = 10000
n_grid = 100

# Create polar grid.
theta = np.linspace(0, 2*np.pi, m_grid)
r = np.linspace(0, 1, n_grid)
theta_grid, r_grid = np.meshgrid(theta, r)

# Define structure regions.
regions = np.zeros((n_grid, m_grid))
regions[r_grid <= 0.35] = 1   # OAR.
regions[(r_grid >= 0.5) & (r_grid <= 0.65) & (theta_grid <= np.pi/2)] = 2   # PTV.
K = np.unique(regions).size

# Plot regions.
fig, ax = plt.subplots(1, 1, subplot_kw = dict(projection = "polar"))
ctf = ax.contourf(theta_grid, r_grid, regions, cmap = "jet")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_theta_zero_location("N")
plt.colorbar(ctf)

# Plot beams.
beam_angles = np.linspace(0, np.pi, n+1)[:-1]
arr = np.ones((2*n, 2))
arr[0::2,0] = beam_angles
arr[1::2,0] = beam_angles + np.pi
segments = np.split(arr, n)
lc = LineCollection(segments, color = "grey")
ax.add_collection(lc)
plt.show()

A = line_integral_mat(theta_grid, regions, beam_angles = beam_angles, atol = 1e-4)
