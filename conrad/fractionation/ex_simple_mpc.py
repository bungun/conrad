import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from plot_utils import *
from data_utils import line_integral_mat, health_prognosis
from example_utils import simple_structures, simple_colormap
from mpc_funcs import dynamic_treatment, mpc_treatment

np.random.seed(1)

T = 20     # Length of treatment.
n_grid = 1000
offset = 0.01   # Displacement between beams.
n_angle = 10   # Number of angles.
n_bundle = 10   # Number of beams per angle.
n = n_angle*n_bundle   # Total number of beams.

# Display structures on a polar grid.
x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
struct_kw = simple_colormap()
# plot_structures(x_grid, y_grid, regions, **struct_kw)
# plot_structures(x_grid, y_grid, regions, filename = "ex_cardioid_structures.png", **struct_kw)

# Problem data.
K = np.unique(regions).size   # Number of structures.
A, angles, offs_vec = line_integral_mat(regions, angles = n_angle, n_bundle = n_bundle, offset = offset)
# A = np.load("data/A_cardiod_rot_1000x1000-grid_10-angles_10-bundles.npy")
# angles = np.linspace(0, np.pi, n_angle+1)[:-1]
# offs_vec = offset*np.arange(-n_bundle//2, n_bundle//2)
# TODO: Define list of A matrices with slight uncertainty.
A_list = T*[A]

# F = np.diag([1.02, 0.95, 0.90, 0.75])
F = np.diag([1.02, 0.90, 0.75, 0.95])
# G = -np.eye(K)
G = -np.diag([0.01, 0.95, 0.25, 0.0065])
r = np.zeros(K)
h_init = np.array([1] + (K-1)*[0])

# Actual health status transition function.
mu = 0
sigma = 0.025
h_noise = mu + sigma*np.random.randn(T,K)
# health_map = lambda h,t: h
def health_map(h,t):
	h_jitter = h + h_noise[t]
	h_jitter[0] = np.maximum(h_jitter[0], 0)     # PTV: h_t >= 0.
	h_jitter[1:] = np.minimum(h_jitter[1:], 0)   # OAR: h_t <= 0.
	return h_jitter

# Health prognosis.
h_prog = health_prognosis(h_init, T, F, r_list = r, health_map = health_map)
curves = {"Untreated": h_prog}

# Penalty function.
rx_health_weights = [K*[1], K*[1]]
rx_health_goal = np.zeros(K)
rx_dose_weights = K*[1]
rx_dose_goal = np.zeros(K)
rx_dose_goal[0] = 10
# rx_dose_goal = np.full((K,), 0.25)
patient_rx = {"dose_goal": rx_dose_goal, "dose_weights": rx_dose_weights, \
			  "health_goal": rx_health_goal, "health_weights": rx_health_weights}

# Beam constraints.
beam_upper = np.full((T,n), 0.0015)
patient_rx["beam_constrs"] = {"upper": beam_upper}

# Dose constraints.
dose_lower = np.zeros((T,K))
dose_upper = np.full((T,K), 25)   # Upper bound on doses.
patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

# Health constraints.
health_lower = np.zeros((T,K))
health_upper = np.zeros((T,K))
# health_lower[:,1:] = -2.5    # Lower bound on OARs.
health_lower[:,1] = -0.5
health_lower[:,2] = -0.75
health_lower[:,3] = -1.0
health_upper[:15,0] = 1.5    # Upper bound on PTV for t = 1,...,15.
health_upper[15:,0] = 0.05   # Upper bound on PTV for t = 16,...,20.
patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}

# Dynamic treatment.
res_dynamic = dynamic_treatment(A_list, F, G, r, h_init, patient_rx, health_map = health_map, solver = "MOSEK")
print("Dynamic Treatment")
print("Status:", res_dynamic["status"])
print("Objective:", res_dynamic["obj"])
print("Solve Time:", res_dynamic["solve_time"])

# Plot dynamic beam, health, and treatment curves.
# Set beam colors on logarithmic scale.
b_min = np.min(res_dynamic["beams"][res_dynamic["beams"] > 0])
b_max = np.max(res_dynamic["beams"])
lc_norm = LogNorm(vmin = b_min, vmax = b_max)

plot_beams(res_dynamic["beams"], angles = angles, offsets = offs_vec, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
			structures = (x_grid, y_grid, regions), struct_kw = struct_kw)
plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper))
plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper))

# plot_beams(res_dynamic["beams"], angles = angles, offsets = offs_vec, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
#			structures = (x_grid, y_grid, regions), struct_kw = struct_kw, filename = "ex_noisy_dyn_beams.png")
# plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper), filename = "ex_noisy_dyn_health.png")
# plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper), filename = "ex_noisy_dyn_doses.png")

# Modify health constraints for MPC.
health_upper_mpc = health_upper.copy()
health_upper_mpc[:15,0] = np.linspace(1.5, 0.05, num = 15)
patient_rx_mpc = patient_rx.copy()
patient_rx_mpc["health_constrs"]["upper"] = health_upper_mpc

# Dynamic treatment with MPC.
res_mpc = mpc_treatment(A_list, F, G, r, h_init, patient_rx_mpc, health_map = health_map, solver = "MOSEK")
print("\nMPC Treatment")
print("Status:", res_mpc["status"])
print("Objective:", res_mpc["obj"])
print("Solve Time:", res_mpc["solve_time"])

# Plot dynamic MPC beam, health, and treatment curves.
# Set beam colors on logarithmic scale.
b_min = np.min(res_mpc["beams"][res_mpc["beams"] > 0])
b_max = np.max(res_mpc["beams"])
lc_norm = LogNorm(vmin = b_min, vmax = b_max)

plot_beams(res_mpc["beams"], angles = angles, offsets = offs_vec, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
			structures = (x_grid, y_grid, regions), struct_kw = struct_kw)
plot_health(res_mpc["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper))
plot_treatment(res_mpc["doses"], stepsize = 10, bounds = (dose_lower, dose_upper))

# plot_beams(res_mpc["beams"], angles = angles, offsets = offs_vec, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
#		  structures = (x_grid, y_grid, regions), struct_kw = struct_kw, filename = "ex_noisy_mpc_beams.png")
# plot_health(res_mpc["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper_mpc), filename = "ex_noisy_mpc_health.png")
# plot_treatment(res_mpc["doses"], stepsize = 10, bounds = (dose_lower, dose_upper), filename = "ex_noisy_mpc_doses.png")
