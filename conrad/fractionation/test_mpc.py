import numpy as np
from mpc_funcs import *
from data_utils import *
from plot_utils import *

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
rx_dose = [66] + (K-1)*[0]
w_under = K*[1]
w_over = [2] + (K-1)*[1]
rx_weights = [w_under, w_over]

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
h_prog = health_prognosis_gen(F, h_init, T, health_map = health_map)
curves = {"Untreated": h_prog}

# Single treatment.
res_single = single_treatment(A, rx_dose, rx_weights, solver = "MOSEK")
print("Single Treatment")
print("Status:", res_single["status"])
print("Objective:", res_single["obj"])

# Dynamic treatment.
res_dynamic = dynamic_treat_recover(A_list, F, G, h_init, rx_dose, rx_weights, T_recov, health_map = health_map, solver = "MOSEK")
print("\nDynamic Treatment")
print("Status:", res_dynamic["status"])
print("Objective:", res_dynamic["obj"])

# Plot dynamic health and treatment curves.
plot_health(res_dynamic["health"], curves = curves, stepsize = 10, T_treat = T_treat)
plot_treatment(res_dynamic["doses"], stepsize = 10, T_treat = T_treat)

# Dynamic treatment with MPC.
res_mpc = mpc_treat_recover(A_list, F, G, h_init, rx_dose, rx_weights, T_recov, health_map = health_map, solver = "MOSEK")
print("\nMPC Treatment")
print("Status:", res_mpc["status"])
print("Objective:", res_mpc["obj"])

# Plot dynamic MPC health and treatment curves.
plot_health(res_mpc["health"], curves = curves, stepsize = 10, T_treat = T_treat)
plot_treatment(res_mpc["doses"], stepsize = 10, T_treat = T_treat)
