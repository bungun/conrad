import numpy as np
import cvxpy
from cvxpy import *
from data_utils import *
from plot_utils import *

# Health status uncertainty.
def health_noise(shape, mu = 0, sigma = 1):
	return mu + sigma*np.random.randn(*shape)

# Growth rates.          x_0   x_1   x_2   x_3
growth_mat = np.array([[1.02,    0,    0,    0],   # x_0 = PTV
					   [   0, 0.95,    0,    0],   # x_1 = OAR_1
					   [   0,    0, 0.90,    0],   # x_2 = OAR_2
					   [   0,    0,    0, 0.75]])  # x_3 = OAR_3

# growth_mat = np.array([[  1.02,    0,    0,    0],   # x_0 = PTV
# 				         [-0.075, 0.95,    0,    0],   # x_1 = OAR_1
#   			         [ -0.05,    0, 0.90,    0],   # x_2 = OAR_2
# 				         [ -0.02,    0,    0, 0.75]])  # x_3 = OAR_3
					   
# Damage rates.		     u_0   u_1   u_2
damage_mat = np.array([[0.15, 0.30, -0.05],   # x_0 = PTV
					   [0.10, 0.25, -0.05],   # x_1 = OAR_1
					   [0.10, 0.25, -0.05],   # x_2 = OAR_2
					   [0.10, 0.25, -0.05]])  # x_3 = OAR_3

# Problem data.
T_treat = 14   # Treatment period.
T_recov = 20   # Recovery period.
T = T_treat + T_recov  # Number of time periods.
m = growth_mat.shape[0]   # Number of health measures.
n = damage_mat.shape[1]   # Number of treatment plans.

# Constraints.
OAR_start = 0
OAR_min = np.full((T+1,m-1), -0.25)
# OAR_min = np.full((T+1,m-1), -0.1)
PTV_start = 0.25
PTV_worst = 0.5
PTV_end = 0.001
PTV_min = 0
PTV_max = np.array([PTV_start] + T_treat*[PTV_worst] + T_recov*[PTV_end])
x_init = np.array([PTV_start] + (m-1)*[OAR_start])

# Construct problem.
x = Variable((T+1,m))   # Health (state).
u = Variable((T+1,n))   # Treatment (control).
d_lo = Variable((T+1,m))   # Slack variable on health.
d_hi = Variable((T+1,m))

# Split into PTVs and OARs.
x_OAR = x[:,1:]
x_PTV = x[:,0]

d_lo_OAR = d_lo[:,1:]
d_lo_PTV = d_lo[:,0]
d_hi_OAR = d_hi[:,1:]
d_hi_PTV = d_hi[:,0]

# Initialize values.
mu = 0            # Health status noise parameters.
sigma = 0.0025
lam_OAR = 1       # Slack regularization coefficients.
lam_PTV = 10
x_final = np.zeros((T+1,m))
u_final = np.zeros((T+1,n))
w_final = np.zeros((T,m))

eps = health_noise((m,), mu, sigma)
x_cur = x_init + eps
x_final[0] = x_cur

# Model predictive control.
for t_start in range(T-1):
	# Noise in health measurement.
	w = health_noise((T,m), mu, sigma)
	
	# Construct problem.
	# Minimize \sum_{t=t_start}^T ||u_t||_2 + ||d_t||_1
	# subject to x_{t+1} = growth*x_t - damage*u_t + w_t.
	reg_OAR = lam_OAR*norm1(vstack([d_lo_OAR[t_start:], d_hi_OAR[t_start:]]))
	reg_PTV = lam_PTV*norm1(vstack([d_lo_PTV[t_start:], d_hi_PTV[t_start:]]))
	obj = pnorm(u[t_start:],2) + reg_OAR + reg_PTV
	constr = [x[t+1] == growth_mat*x[t] - damage_mat*u[t] + w[t] for t in range(t_start,T)]
	constr += [x[t_start] == x_cur, x_PTV[t_start:] <= PTV_max[t_start:] + d_hi_PTV[t_start:],
			   x_PTV[t_start:] >= PTV_min - d_lo_PTV[t_start:], 
			   x_OAR[t_start:] >= OAR_min[t_start:] - d_lo_OAR[t_start:],
			   u[t_start:] >= 0, u[T_treat:] == 0]
	
	# Solve problem.
	prob = Problem(Minimize(obj), constr)
	prob.solve()
	print("Start Time:", t_start)
	print("Status:", prob.status)
	print("Objective:", prob.value)
	
	# Update actual health status and treatment.
	eps = health_noise((m,), mu, sigma)
	x_cur = x[t_start+1].value + eps
	x_final[t_start+1] = x_cur
	u_final[t_start] = u[t_start].value
	w_final[t_start] = eps

# Plot health and treatment curves.
u_final[T] = u_final[T-1]
x_prog = health_prognosis(x_init, growth_mat, T, w_noise = w_final)
plot_health(x_final, {"Untreated": x_prog}, stepsize = 10, T_treat = T_treat)
plot_treatment(u_final, stepsize = 10, T_treat = T_treat)
# plot_health(x_final, {"Untreated": x_prog}, stepsize = 10, T_treat = T_treat, filename = "mpc_simple_health.png")
# plot_treatment(u_final, stepsize = 10, T_treat = T_treat, filename = "mpc_simple_treatment.png")
