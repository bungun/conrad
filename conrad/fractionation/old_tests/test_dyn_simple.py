import numpy as np
from mpc_funcs import *
from data_utils import beam_to_dose_block, health_prognosis
from plot_utils import plot_health, plot_treatment

# Problem data.
T_treat = 14
T_recov = 20
T = T_treat + T_recov

# Dynamics matrices.
A = np.array([[0.15, 0.30, -0.05],   # x_0 = PTV
			  [0.10, 0.25, -0.05],   # x_1 = OAR_1
			  [0.10, 0.25, -0.05],   # x_2 = OAR_2
			  [0.10, 0.25, -0.05]])  # x_3 = OAR_3
K = A.shape[0]

F = np.diag([1.02, 0.95, 0.90, 0.75])
G = -np.eye(K)
A_list = T_treat*[A]
h_init = np.array([0.25] + (K-1)*[0])

# Prescription.
# rx_dose = K*[0]
rx_dose = np.zeros((T_treat,K))
w_under = K*[1]
w_over = K*[1]
rx_dose_weights = [w_under, w_over]
rx_health_weights = K*[1]
patient_rx = {"dose": rx_dose, "dose_weights": rx_dose_weights, "health_weights": rx_health_weights}

# Health prognosis.
health_map = lambda h,t: h
# h_prog = health_prognosis(h_init, F, T, w_noise = h_noise)
h_prog = health_prognosis(F, h_init, T, health_map = health_map)
curves = {"Untreated": h_prog}

# Health constraints during treatment.
health_lower = np.full((T_treat,K), -np.inf)
health_upper = np.full((T_treat,K), np.inf)
health_lower[:,0] = 0
health_lower[:,1:] = -0.25
health_upper[:,0] = np.full(T_treat, 0.5)
patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}

# Health constraints during recovery.
recov_lower = np.full((T_recov,K), -np.inf)
recov_upper = np.full((T_recov,K), np.inf)
recov_lower[:,0] = 0
recov_lower[:,1:] = -0.25
recov_upper[:,0] = 0.001
patient_rx["recov_constrs"] = {"lower": recov_lower, "upper": recov_upper}

# Dynamic treatment.
res_dynamic = dynamic_treatment(A_list, F, G, h_init, patient_rx, T_recov, health_map = health_map, solver = "MOSEK")
print("Dynamic Treatment")
print("Status:", res_dynamic["status"])
print("Objective:", res_dynamic["obj"])

# Plot dynamic health and treatment curves.
plot_health(res_dynamic["health"], curves = curves, stepsize = 10, T_treat = T_treat)
# plot_treatment(res_dynamic["doses"], stepsize = 10, T_treat = T_treat)
plot_treatment(res_dynamic["beams"], stepsize = 10, T_treat = T_treat)
