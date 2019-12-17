import numpy as np
from admm_funcs import *
from mpc_funcs import dynamic_treatment
from data_utils import beam_to_dose_block, health_prognosis
from plot_utils import plot_health, plot_treatment, plot_residuals

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

# Prescription.
rx_dose = np.array([66] + (K-1)*[0])
w_under = K*[1]
w_over = [2] + (K-1)*[1]
rx_dose_weights = [w_under, w_over]
rx_health_weights = K*[1]
patient_rx = {"dose": rx_dose, "dose_weights": rx_dose_weights, "health_weights": rx_health_weights}

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
A_list = T_treat*[A]
h_init = np.array([0.25] + (K-1)*[0])

# Actual health status transition function.
mu = 0
sigma = 0.005
h_noise = mu + sigma*np.random.randn(T,K)
health_map = lambda h,t: h + h_noise[t]

# Health prognosis.
# h_prog = health_prognosis(h_init, F, T, w_noise = h_noise)
h_prog = health_prognosis(F, h_init, T, health_map = health_map)
curves = {"Untreated": h_prog}

# Dose constraints.
# dose_lower = np.full((T_treat,K), -np.inf)
# dose_upper = np.full((T_treat,K), np.inf)
# dose_lower[:,0] = np.concatenate([np.full(5, 25), np.full(15, -np.inf)])
# dose_upper[:,0] = 75
# dose_upper[:,1:] = 25
# patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

# Prescription per period.
patient_rx["dose"] = np.tile(rx_dose/T_treat, (T_treat,1))

# Dynamic treatment.
res_dynamic = dynamic_treatment(A_list, F, G, h_init, patient_rx, T_recov, health_map = health_map, solver = "MOSEK")
print("Dynamic Treatment")
print("Status:", res_dynamic["status"])
print("Objective:", res_dynamic["obj"])

# Plot dynamic health and treatment curves.
plot_health(res_dynamic["health"], curves = curves, stepsize = 10, T_treat = T_treat, filename = "simple_dyn_health.png")
plot_treatment(res_dynamic["doses"], stepsize = 10, T_treat = T_treat, filename = "simple_dyn_treatment.png")

# Dynamic treatment with ADMM.
res_dynamic = dynamic_treatment_admm(A_list, F, G, h_init, patient_rx, T_recov, health_map = health_map, rho = 0.5, max_iter = 1000, solver = "MOSEK")
print("\nDynamic Treatment with ADMM")
print("Status:", res_dynamic["status"])
print("Objective:", res_dynamic["obj"])
print("Iterations:", res_dynamic["num_iters"])

# Plot dynamic ADMM health and treatment curves.
plot_residuals(res_dynamic["primal"], res_dynamic["dual"], semilogy = True, filename = "simple_admm_residuals.png")
plot_health(res_dynamic["health"], curves = curves, stepsize = 10, T_treat = T_treat, filename = "simple_admm_health.png")
plot_treatment(res_dynamic["doses"], stepsize = 10, T_treat = T_treat, filename = "simple_admm_treatment.png")

# Dynamic treatment with MPC.
# res_mpc = mpc_treatment_admm(A_list, F, G, h_init, patient_rx, T_recov, health_map = health_map, solver = "MOSEK")
# print("\nMPC Treatment")
# print("Status:", res_mpc["status"])
# print("Objective:", res_mpc["obj"])

# Plot dynamic MPC health and treatment curves.
# plot_health(res_mpc["health"], curves = curves, stepsize = 10, T_treat = T_treat)
# plot_treatment(res_mpc["doses"], stepsize = 10, T_treat = T_treat)
