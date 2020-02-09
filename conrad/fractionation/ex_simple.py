import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from plot_utils import *
<<<<<<< Updated upstream
from data_utils import line_integral_mat, circle, ellipse
=======
from data_utils import line_integral_mat, health_prognosis
>>>>>>> Stashed changes
from mpc_funcs import dynamic_treatment, mpc_treatment

n = 1000   # Number of beams.
T = 20     # Length of treatment.
m_grid = 10000
n_grid = 500

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

# Create colormap for structures.
K = np.unique(regions).size   # Number of structures.
struct_cmap = ListedColormap(['red', 'blue', 'green', 'white'])
struct_bounds = np.arange(K+1) - 0.5
# struct_cmap = truncate_cmap(plt.cm.rainbow, 0, 0.1*K, n = K)
# struct_bounds = np.linspace(0, K, K+1)
struct_norm = BoundaryNorm(struct_bounds, struct_cmap.N)
struct_kw = {"cmap": struct_cmap, "norm": struct_norm, "alpha": 0.5}
# plot_structures(theta_grid, r_grid, regions, **struct_kw)

# Problem data.
A = line_integral_mat(theta_grid, regions, beam_angles = n, atol = 1e-4)
A_list = T*[A]

# F = np.diag([1.02, 0.95, 0.90, 0.75])
F = np.diag([1.02, 0.90, 0.75, 0.95])
G = -np.eye(K)
r = np.zeros(K)
h_init = np.array([1] + (K-1)*[0])

# Health prognosis.
h_prog = health_prognosis(h_init, T, F, r_list = r)
curves = {"Untreated": h_prog}

# Penalty function.
rx_health_weights = [K*[1], K*[1]]
rx_health_goal = np.zeros(K)
rx_dose_weights = K*[1]
# rx_dose_goal = np.zeros(K)
rx_dose_goal = np.full((K,), 0.25)
patient_rx = {"dose_goal": rx_dose_goal, "dose_weights": rx_dose_weights, \
			  "health_goal": rx_health_goal, "health_weights": rx_health_weights}

# Beam constraints.
beam_upper = np.full((T,n), 0.5)
patient_rx["beam_constrs"] = {"upper": beam_upper}

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
print("Solve Time:", res_dynamic["solve_time"])

# Plot dynamic beam, health, and treatment curves.
# TODO: Modify beam colorbar to show changes in intensity.
plot_beams(res_dynamic["beams"], stepsize = 4, cmap = transp_cmap(plt.cm.Reds), \
			structures = (theta_grid, r_grid, regions), struct_kw = struct_kw)
plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper))
plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper))

# Dynamic treatment with MPC.
# res_mpc = mpc_treatment(A_list, F, G, r, h_init, patient_rx, solver = "MOSEK")
# print("\nMPC Treatment")
# print("Status:", res_mpc["status"])
# print("Objective:", res_mpc["obj"])
# print("Solve Time:", res_mpc["solve_time"])

# Plot dynamic MPC beam, health, and treatment curves.
# plot_beams(res_mpc["beams"], stepsize = 4, cmap = transp_cmap(plt.cm.Reds), \
# 			  structures = (theta_grid, r_grid, regions), struct_kw = struct_kw)
# plot_health(res_mpc["health"], stepsize = 10, bounds = (health_lower, health_upper))
# plot_treatment(res_mpc["doses"], stepsize = 10, bounds = (dose_lower, dose_upper))
