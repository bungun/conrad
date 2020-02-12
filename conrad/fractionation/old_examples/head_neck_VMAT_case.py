import numpy as np
from mpc_funcs import *
from data_utils import beam_to_dose_block, health_prognosis
from plot_utils import plot_health, plot_treatment

# Beam-to-dose matrix.
A_full = np.load("/home/anqi/Documents/data/MCdose/A_block_sorted.npy")
A_counts = np.load("/home/anqi/Documents/data/MCdose/countsByStructure.npy")
indices = np.cumsum(A_counts)[:-1]
A_mean = beam_to_dose_block(A_full, indices)

# Problem data.
T_treat = 20
T_recov = 14
T = T_treat + T_recov
K, n = A_mean.shape
A_list = T_treat*[A_mean]

# Initial health.
PTV_health = [0.25, 0.175, 0.15]
OAR_health = (K-3)*[0]
h_init = np.array(PTV_health + OAR_health)

# Dynamics matrices.
PTV_rate = [1.02, 1.01, 1.01]
OAR_rate = np.random.uniform(0.15, 0.95, K-3).tolist()
F = np.diag(PTV_rate + OAR_rate)
G = -np.eye(K)

# Prescription.
patient_rx = {}
dose_total = np.array([66, 60, 60] + (K-3)*[0])
patient_rx["dose"] = np.tile(dose_total/T_treat, (T_treat,1))
patient_rx["dose_weights"] = [K*[1], K*[1]]
patient_rx["health_weights"] = K*[1]

# Health prognosis.
h_prog = health_prognosis(F, h_init, T)
curves = {"Untreated": h_prog}

# Health constraints during treatment.
health_lower = np.full((T_treat,K), -np.inf)
health_upper = np.full((T_treat,K), np.inf)
health_lower[:,0] = 0
health_lower[:,1:] = -0.1
health_upper[:,0] = 0.5
patient_rx["health_constrs"] = {"lower": health_lower, "upper": health_upper}

# Health constraints during recovery.
recov_lower = np.full((T_recov,K), -np.inf)
recov_upper = np.full((T_recov,K), np.inf)
recov_lower[:,0] = 0
recov_lower[:,1:] = -0.1
recov_upper[:,0] = 0.001
patient_rx["recov_constrs"] = {"lower": recov_lower, "upper": recov_upper}

# Dynamic treatment with MPC.
res_mpc = mpc_treatment(A_list, F, G, h_init, patient_rx, T_recov, solver = "MOSEK")
print("Status:", res_mpc["status"])
print("Objective:", res_mpc["obj"])
print("Solve Time:", res_mpc["solve_time"])

# Plot dynamic MPC health and dose curves.
plot_health(res_mpc["health"], curves = curves, stepsize = 10, T_treat = T_treat)
plot_treatment(res_mpc["doses"], stepsize = 10, T_treat = T_treat)
# plot_health(res_mpc["health"], curves = curves, stepsize = 10, T_treat = T_treat, filename = "head_neck_vmat_health.png")
# plot_treatment(res_mpc["doses"], stepsize = 10, T_treat = T_treat, filename = "head_neck_vmat_doses.png")
