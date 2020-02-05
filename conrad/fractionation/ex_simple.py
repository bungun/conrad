import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from plot_utils import *
from data_utils import line_integral_mat
from mpc_funcs import dynamic_treatment, mpc_treatment

n = 10   # Number of beams.
T = 20    # Length of treatment.
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
# Body voxels (k = 3).
regions = np.full((n_grid, m_grid), 3)

# PTV (k = 0).
regions[(r_grid >= 0.5) & (r_grid <= 0.65) & (theta_grid <= np.pi/2)] = 0

# OAR (k = 1).
regions[r_grid <= 0.35] = 1

# OAR (k = 2).
x_grid = r_grid*np.cos(theta_grid)
y_grid = r_grid*np.sin(theta_grid)
regions[ellipse(x_grid, y_grid, (x0, y0), (x_width, y_width), np.pi/6) <= 0] = 2

# Create colormap for regions.
K = np.unique(regions).size   # Number of structures.
cmap = ListedColormap(['red', 'blue', 'green', 'white'])
bounds = np.arange(K+1) - 0.5
# cmap = truncate_cmap(plt.cm.rainbow, 0, 0.1*K, n = K)
# bounds = np.linspace(0, K, K+1)
norm = BoundaryNorm(bounds, cmap.N)

# Plot regions.
fig, ax = plt.subplots(1, 1, subplot_kw = dict(projection = "polar"))
ctf = ax.contourf(theta_grid, r_grid, regions, cmap = cmap, norm = norm)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_theta_zero_location("N")
plt.colorbar(ctf)
plt.show()

# Define problem.
A = line_integral_mat(theta_grid, regions, beam_angles = n, atol = 1e-4)
A_list = T*[A]

# F = np.diag([1.02, 0.95, 0.90, 0.75])
F = np.diag([1.02, 0.90, 0.75, 0.95])
G = -np.eye(K)
r = np.zeros(K)
h_init = np.array([1] + (K-1)*[0])

# Penalty function.
rx_health_weights = [K*[1], K*[1]]
rx_health_goal = np.zeros(K)
rx_dose_weights = K*[1]
patient_rx = {"dose_weights": rx_dose_weights, "health_goal": rx_health_goal, "health_weights": rx_health_weights}

# Dose constraints.
dose_lower = np.zeros((T,K))
dose_upper = np.full((T,K), 0.5)   # Upper bound on doses.
patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

# Health constraints.
health_lower = np.zeros((T,K))
health_upper = np.zeros((T,K))
health_lower[:,1:] = -2.5    # Lower bound on OARs.
health_upper[:15,0] = 1.5    # Upper bound on PTV for t = 1,...,15.
health_upper[15:,0] = 0.05   # Upper bound on PTV for t = 16,...,20.
patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}

# Dynamic treatment.
res_dynamic = dynamic_treatment(A_list, F, G, r, h_init, patient_rx, solver = "MOSEK")
print("Dynamic Treatment")
print("Status:", res_dynamic["status"])
print("Objective:", res_dynamic["obj"])

# Plot dynamic health and treatment curves.
plot_health(res_dynamic["health"], stepsize = 10, bounds = (health_lower, health_upper))
plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper))
plot_beams(res_dynamic["beams"], stepsize = 2, cmap = transp_cmap(plt.cm.Reds))

# Dynamic treatment with MPC.
# res_mpc = mpc_treatment(A_list, F, G, r, h_init, patient_rx, solver = "MOSEK")
# print("\nMPC Treatment")
# print("Status:", res_mpc["status"])
# print("Objective:", res_mpc["obj"])

# Plot dynamic MPC health and treatment curves.
# plot_health(res_mpc["health"], stepsize = 10, bounds = (health_lower, health_upper))
# plot_treatment(res_mpc["doses"], stepsize = 10, bounds = (dose_lower, dose_upper))
# plot_beams(res_mpc["beams"], stepsize = 2, cmap = transp_cmap(plt.cm.Reds))
