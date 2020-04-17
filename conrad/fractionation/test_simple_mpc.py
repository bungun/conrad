import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from plot_utils import *
from data_utils import line_integral_mat, health_prognosis
from example_utils import simple_structures, simple_colormap
from mpc_funcs import dynamic_treatment, mpc_treatment
from admm_funcs import dynamic_treatment_admm, mpc_treatment_admm

np.random.seed(1)

T = 20           # Length of treatment.
n_grid = 1000
offset = 5       # Displacement between beams (pixels).
n_angle = 20     # Number of angles.
n_bundle = 50    # Number of beams per angle.
n = n_angle*n_bundle   # Total number of beams.

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Display structures.
x_grid, y_grid, regions = simple_structures(n_grid, n_grid)
struct_kw = simple_colormap(one_idx = True)
# plot_structures(x_grid, y_grid, regions, title = "Anatomical Structures", one_idx = True, **struct_kw)
# plot_structures(x_grid, y_grid, regions, one_idx = True, filename = "ex_cardioid5_structures.png", **struct_kw)

# Problem data.
K = np.unique(regions).size   # Number of structures.
A, angles, offs_vec = line_integral_mat(regions, angles = n_angle, n_bundle = n_bundle, offset = offset)
A = A/n_grid
A_list = T*[A]

F = np.diag([1.05, 0.90, 0.75, 0.80, 0.95])
G = -np.diag([0.01, 0.5, 0.25, 0.15, 0.0075])
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

# Health violation.
def health_viol(h, bounds):
	lower, upper = bounds
	viols = np.maximum(h - upper, 0) + np.maximum(lower - h, 0)
	return np.sum(viols)/T

# Health prognosis.
h_prog = health_prognosis(h_init, T, F, r_list = r)
h_curves = [{"h": h_prog, "label": "Untreated", "kwargs": {"color": colors[1]}}]

# Penalty function.
rx_health_weights = [K*[1], K*[1]]
rx_health_goal = np.zeros((T,K))
# rx_dose_weights = K*[1]
rx_dose_weights = [0.01, 1, 1, 1, 0.001]
rx_dose_goal = np.zeros((T,K))
# rx_dose_goal[:15,0] = 25
patient_rx = {"dose_goal": rx_dose_goal, "dose_weights": rx_dose_weights, \
			  "health_goal": rx_health_goal, "health_weights": rx_health_weights}

# Beam constraints.
beam_upper = np.full((T,n), 1.0)
patient_rx["beam_constrs"] = {"upper": beam_upper}

# Dose constraints.
dose_lower = np.zeros((T,K))
# dose_upper = np.full((T,K), 25)   # Upper bound on doses.
dose_upper = np.full((T,K), np.inf)
patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

# Health constraints.
health_lower = np.zeros((T,K))
health_upper = np.zeros((T,K))
# health_lower[:,1:] = -np.inf    # Lower bound on OARs.
# health_lower[:,1] = -0.5
health_lower[:,2] = -0.5
health_lower[:,3] = -0.75   # -0.95
health_lower[:,4] = -1.15   # -1.25
health_upper[:15,0] = 1.5    # Upper bound on PTV for t = 1,...,15.
health_upper[15:,0] = 0.05   # Upper bound on PTV for t = 16,...,20.
patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}

# Dynamic treatment.
res_dynamic = dynamic_treatment(A_list, F, G, r, h_init, patient_rx, health_map = health_map, solver = "MOSEK")
# res_dynamic = dynamic_treatment_admm(A_list, F, G, r, h_init, patient_rx, health_map = health_map, rho = 10, max_iter = 1000, solver = "ECOS", admm_verbose = True)
print("Dynamic Treatment")
print("Status:", res_dynamic["status"])
print("Objective:", res_dynamic["obj"])
print("Solve Time:", res_dynamic["solve_time"])
# print("Iterations:", res_dynamic["num_iters"])

# Calculate actual health constraint violation.
h_viol_dyn = health_viol(res_dynamic["health"][1:], (health_lower, health_upper))
print("Average Health Violation", h_viol_dyn)

# Set beam colors on logarithmic scale.
b_min = np.min(res_dynamic["beams"][res_dynamic["beams"] > 0])
b_max = np.max(res_dynamic["beams"])
lc_norm = LogNorm(vmin = b_min, vmax = b_max)

# Plot dynamic beam, health, and treatment curves.
# plot_residuals(res_dynamic["primal"], res_dynamic["dual"], semilogy = True)
plot_beams(res_dynamic["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
			title = "Beam Intensities vs. Time", one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw)
plot_health(res_dynamic["health"], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), title = "Health Status vs. Time", one_idx = True)
plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper), title = "Treatment Dose vs. Time", one_idx = True)

# Dynamic treatment with MPC.
print("\nStarting MPC algorithm...")
res_mpc = mpc_treatment(A_list, F, G, r, h_init, patient_rx, health_map = health_map, solver = "MOSEK", mpc_verbose = True)
# res_mpc = mpc_treatment_admm(A_list, F, G, r, h_init, patient_rx, health_map = health_map, rho = 10, max_iter = 1000, solver = "ECOS", mpc_verbose = True)
print("\nMPC Treatment")
print("Status:", res_mpc["status"])
print("Objective:", res_mpc["obj"])
print("Solve Time:", res_mpc["solve_time"])
# print("Iterations:", res_mpc["num_iters"])

# Calculate actual health constraint violation.
h_viol_mpc = health_viol(res_mpc["health"][1:], (health_lower, health_upper))
print("Average Health Violation", h_viol_mpc)

# Set beam colors on logarithmic scale.
b_min = np.min(res_mpc["beams"][res_mpc["beams"] > 0])
b_max = np.max(res_mpc["beams"])
lc_norm = LogNorm(vmin = b_min, vmax = b_max)

# Compare one-shot dynamic and MPC treatment results.
d_curves = [{"d": res_dynamic["doses"], "label": "Naive", "kwargs": {"color": colors[0]}}]
h_naive = [{"h": res_dynamic["health"], "label": "Treated (Naive)", "kwargs": {"color": colors[0]}}]
h_curves = h_naive + h_curves

# Plot dynamic MPC beam, health, and treatment curves.
plot_beams(res_mpc["beams"], angles = angles, offsets = offs_vec, n_grid = n_grid, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
			title = "Beam Intensities vs. Time", one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw)
plot_health(res_mpc["health"], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), \
			title = "Health Status vs. Time", label = "Treated (MPC)", color = colors[2], one_idx = True)
plot_treatment(res_mpc["doses"], curves = d_curves, stepsize = 10, bounds = (dose_lower, dose_upper), \
			title = "Treatment Dose vs. Time", label = "MPC", color = colors[2], one_idx = True)

# plot_beams(res_mpc["beams"], angles = angles, offsets = offs_vec, stepsize = 1, cmap = transp_cmap(plt.cm.Reds, upper = 0.5), \
#		  	one_idx = True, structures = (x_grid, y_grid, regions), struct_kw = struct_kw, filename = "ex_noisy2_mpc_admm_beams.png")
# plot_health(res_mpc["health"], curves = h_curves, stepsize = 10, bounds = (health_lower, health_upper), \
#			label = "Treated (MPC)", color = colors[2], one_idx = True, filename = "ex_noisy2_mpc_admm_health.png")
# plot_treatment(res_mpc["doses"], curves = d_curves, stepsize = 10, bounds = (dose_lower, dose_upper), \
#			label = "MPC", color = colors[2], one_idx = True, filename = "ex_noisy_mpc2_admm_doses.png")
