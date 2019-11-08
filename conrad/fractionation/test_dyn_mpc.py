import numpy as np
from data_utils import *
from plot_utils import *
from mpc_funcs import *

# Problem data.
np.random.seed(1)
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
rx_dose = K*[0]
w_under = K*[1]
w_over = K*[1]
rx_weights = [w_under, w_over]
patient_rx = {"dose": rx_dose, "weights": rx_weights}

# Actual health status transition function.
mu = 0
sigma = 0.005
h_noise = mu + sigma*np.random.randn(T,K)
# health_map = lambda h,t: h + h_noise[t]
health_map = lambda h,t: h

# Health prognosis.
# h_prog = health_prognosis(h_init, F, T, w_noise = h_noise)
h_prog = health_prognosis_gen(F, h_init, T, health_map = health_map)
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

# Dynamic treatment with MPC.
res_mpc = mpc_treatment(A_list, F, G, h_init, patient_rx, T_recov, health_map = health_map, solver = "ECOS")
print("MPC Treatment")
print("Status:", res_mpc["status"])
print("Objective:", res_mpc["obj"])

# Plot dynamic MPC health and treatment curves.
plot_health(res_mpc["health"], curves = curves, stepsize = 10, T_treat = T_treat)
plot_treatment(res_mpc["doses"], stepsize = 10, T_treat = T_treat)
