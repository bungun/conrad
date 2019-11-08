import numpy as np
from mpc_funcs import *
from data_utils import beam_to_dose_block
from plot_utils import plot_health, plot_treatment

np.random.seed(1)

# Problem data.
# m = 10000
# n = 1000
# K = 10
m = 1000
n = 100
K = 4
T_treat = 14
T_recov = 20
T = T_treat + T_recov

# Beam-to-dose matrix.
A_full = np.abs(100 + 10*np.random.randn(m,n))
A = beam_to_dose_block(A_full, K)
A_list = T_treat*[A]

# Dynamics matrices.
F = np.diag([1.02] + np.random.uniform(0.15, 0.95, K-1).tolist())
G = -np.eye(K)
h_init = np.array([0.25] + (K-1)*[0])

# Prescription.
patient_rx = {}
patient_rx["dose"] = [0.66] + (K-1)*[0]
patient_rx["weights"] = [K*[1], [2] + (K-1)*[1]]

# Dose constraints.
dose_lower = np.full((T_treat,K), 0)
dose_upper = np.full((T_treat,K), np.inf)
dose_upper[:,0] = 0.1
dose_upper[:,1:] = 0.05
patient_rx["dose_constrs"] = {"lower": dose_lower, "upper": dose_upper}

# Health prognosis.
h_prog = health_prognosis_gen(F, h_init, T)
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
res_mpc = mpc_treatment(A_list, F, G, h_init, patient_rx, T_recov, solver = "ECOS")
print("Status:", res_mpc["status"])
print("Objective:", res_mpc["obj"])
print("Solve Time:", res_mpc["solve_time"])

# Plot dynamic MPC health and dose curves.
plot_health(res_mpc["health"], curves = curves, stepsize = 10, T_treat = T_treat)
plot_treatment(res_mpc["doses"], stepsize = 10, T_treat = T_treat)
# plot_health(x_final, {"Untreated": x_prog}, stepsize = 10, T_treat = T_treat, filename = "mpc_clinical_health.png")
# plot_treatment(u_final, stepsize = 10, T_treat = T_treat, filename = "mpc_clinical_doses.png")
