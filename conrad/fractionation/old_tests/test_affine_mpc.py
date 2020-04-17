import numpy as np
from mpc_funcs import *
from data_utils import beam_to_dose_block, health_prognosis
from plot_utils import plot_health, plot_treatment, plot_beams, transp_cmap

# Problem data.
# m = 10000
# n = 1000
# K = 10
m = 1000
n = 100
K = 4
T_treat = 20
T_recov = 14
T = T_treat + T_recov

# Beam-to-dose matrix.
A_full = np.abs(100 + 10*np.random.randn(m,n))
A_blocks = np.split(A_full, K)
# A_rows = [np.sum(block, axis = 0) for block in A_blocks]
A_rows = [np.mean(block, axis = 0) for block in A_blocks]
A = np.row_stack(A_rows)

# Dynamics matrices.
# F = np.diag([1.02, 0.95, 0.90, 0.85, 0.70, 0.65, 0.50, 0.25, 0.20, 0.15])
F = np.diag([1.02, 0.95, 0.90, 0.75])
G = -np.eye(K)
r = np.zeros(K)
A_list = T_treat*[A]
h_init = np.array([0.25] + (K-1)*[0])

# Actual health status transition function.
mu = 0
sigma = 0.005
h_noise = mu + sigma*np.random.randn(T,K)
health_map = lambda h,t: h + h_noise[t]

# Health prognosis.
h_prog = health_prognosis(h_init, T, F, r_list = r, health_map = health_map)
curves = {"Untreated": h_prog}

# Patient prescription.
D_target = 0.005
D_other = 0.025
H_target = 0
H_other = -0.2
eps = 1e-4

# Penalty function.
# TODO: Allow dose/health weights to vary over time.
w_under = K*[0]
w_over = [1] + (K-1)*[0]
rx_health_weights = [w_under, w_over]
rx_health_goal = np.concatenate(([H_target], h_init[1:]))
rx_dose_weights = K*[0]
patient_rx = {"dose_weights": rx_dose_weights, "health_goal": rx_health_goal, "health_weights": rx_health_weights}

# Dose constraints.
dose_lower = np.full((T_treat,K), -np.inf)
dose_upper = np.full((T_treat,K), np.inf)
dose_lower[:,0] = np.concatenate([np.full(5, D_target), np.full(15, -np.inf)])
dose_upper[:,1:] = D_other
patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

# Health constraints.
health_lower = np.full((T_treat,K), -np.inf)
health_upper = np.full((T_treat,K), np.inf)
health_lower[:,1:] = H_other
# health_lower[T_treat-1,0] = H_target - eps
# health_upper[T_treat-1,0] = H_target + eps
health_upper[T_treat-1,0] = H_target
patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}

# Dynamic treatment.
res_dynamic = dynamic_treatment(A_list, F, G, r, h_init, patient_rx, T_recov, health_map = health_map, solver = "MOSEK")
print("Dynamic Treatment")
print("Status:", res_dynamic["status"])
print("Objective:", res_dynamic["obj"])

# Plot dynamic health and treatment curves.
plot_health(res_dynamic["health"], curves = curves, stepsize = 10, T_treat = T_treat)
plot_treatment(res_dynamic["doses"], stepsize = 10, T_treat = T_treat)
plot_beams(res_dynamic["beams"], stepsize = 10)

# Dynamic treatment with MPC.
res_mpc = mpc_treatment(A_list, F, G, r, h_init, patient_rx, T_recov, health_map = health_map, solver = "MOSEK")
print("\nMPC Treatment")
print("Status:", res_mpc["status"])
print("Objective:", res_mpc["obj"])

# Plot dynamic MPC health and treatment curves.
plot_health(res_mpc["health"], curves = curves, stepsize = 10, T_treat = T_treat)
plot_treatment(res_mpc["doses"], stepsize = 10, T_treat = T_treat)
plot_beams(res_mpc["beams"], stepsize = 10)
