import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from plot_utils import *
from data_utils import line_integral_mat, health_prognosis
from example_utils import simple_structures, simple_colormap
from mpc_funcs import dynamic_treatment
from admm_funcs import dynamic_treatment_admm

n = 1000   # Number of beams.
T = 20     # Length of treatment.
m_grid = 10000
n_grid = 500

# Display structures on a polar grid.
theta_grid, r_grid, regions = simple_structures(m_grid, n_grid)
struct_kw = simple_colormap()
# plot_structures(theta_grid, r_grid, regions, **struct_kw)

# Problem data.
K = np.unique(regions).size   # Number of structures.
# A = line_integral_mat(theta_grid, regions, beam_angles = n, atol = 1e-4)
A = np.load("data/A_simple_10000x500-grid_1000-beams.npy")
# A = np.load("data/A_simple_10000x500-grid_10-beams.npy")
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
res_dynamic = dynamic_treatment(A_list, F, G, r, h_init, patient_rx, solver = "OSQP")
# res_dynamic = dynamic_treatment_admm(A_list, F, G, r, h_init, patient_rx, rho = 5, max_iter = 1000, solver = "OSQP", admm_verbose = True)
print("Dynamic Treatment")
print("Status:", res_dynamic["status"])
print("Objective:", res_dynamic["obj"])
print("Solve Time:", res_dynamic["solve_time"])
print("Iterations:", res_dynamic["num_iters"])

# Plot dynamic beam, health, and treatment curves.
# Set beam colors on logarithmic scale.
b_min = np.min(res_dynamic["beams"][res_dynamic["beams"] > 0])
b_max = np.max(res_dynamic["beams"])
lc_norm = LogNorm(vmin = b_min, vmax = b_max)

plot_residuals(res_dynamic["primal"], res_dynamic["dual"], semilogy = True)
plot_beams(res_dynamic["beams"], stepsize = 1, cmap = transp_cmap(plt.cm.Reds), norm = lc_norm, \
			structures = (theta_grid, r_grid, regions), struct_kw = struct_kw)
plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper))
plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper))

# plot_residuals(res_dynamic["primal"], res_dynamic["dual"], semilogy = True, filename = "ex_simple_admm_dyn_beams.png")
# plot_beams(res_dynamic["beams"], stepsize = 1, cmap = transp_cmap(plt.cm.Reds), norm = lc_norm, \
#  			structures = (theta_grid, r_grid, regions), struct_kw = struct_kw, filename = "ex_simple_admm_dyn_beams.png")
# plot_health(res_dynamic["health"], curves = curves, stepsize = 10, bounds = (health_lower, health_upper), filename = "ex_simple_admm_dyn_health.png")
# plot_treatment(res_dynamic["doses"], stepsize = 10, bounds = (dose_lower, dose_upper), filename = "ex_simple_admm_dyn_doses.png")
