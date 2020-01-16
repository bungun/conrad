import numpy as np
from ccp_funcs import *
from data_utils import beam_to_dose_block
from plot_utils import plot_health, plot_treatment

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
alphas = np.array([1, 2, 3, 4])
betas = np.array([4, 3, 2, 1])
A_list = T_treat*[A]
h_init = np.array([0.25] + (K-1)*[0])

# Actual health status transition function.
mu = 0
sigma = 0.005
h_noise = mu + sigma*np.random.randn(T,K)
health_map = lambda h,t: h + h_noise[t]

# Health prognosis.
h_prog = bed_health_prog(h_init, T, alphas, betas, health_map = health_map)
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
# patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}

# Dynamic treatment using CCP.
res_dynamic = bed_ccp_dyn_treat(A_list, alphas, betas, h_init, patient_rx, T_recov, health_map = health_map, solver = "MOSEK")
print("Dynamic Treatment using CCP")
print("Status:", res_dynamic["status"])
print("Objective:", res_dynamic["obj"])
print("Solve Time:", res_dynamic["solve_time"])
print("Iterations:", res_dynamic["num_iters"])

# Plot dynamic health and treatment curves.
plot_health(res_dynamic["health"], curves = curves, stepsize = 10, T_treat = T_treat)
plot_treatment(res_dynamic["doses"], stepsize = 10, T_treat = T_treat)
