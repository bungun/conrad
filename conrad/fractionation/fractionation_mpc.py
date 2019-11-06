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
#   				     [-0.075, 0.95,    0,    0],   # x_1 = OAR_1
#   				     [ -0.05,    0, 0.90,    0],   # x_2 = OAR_2
#   				     [ -0.02,    0,    0, 0.75]])  # x_3 = OAR_3
					   
# Damage rates.		     u_0   u_1    u_2
damage_mat = np.array([[0.15, 0.30, -0.05],   # x_0 = PTV
					   [0.10, 0.25, -0.05],   # x_1 = OAR_1
					   [0.10, 0.25, -0.05],   # x_2 = OAR_2
					   [0.10, 0.25, -0.05]])  # x_3 = OAR_3

# Problem data.
np.random.seed(1)
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

# Split into PTVs and OARs.
x_OAR = x[:,1:]
x_PTV = x[:,0]

# Initialize values.
mu = 0            # Health status noise parameters.
sigma = 0.005
x_final = np.zeros((T+1,m))
u_final = np.zeros((T+1,n))
w_final = np.zeros((T,m))

w_true = health_noise((m,), mu, sigma)
x_cur = x_init + w_true
x_final[0] = x_cur

# Model predictive control.
for t_s in range(T_treat):
	# Construct problem.
	# Minimize \sum_{t=t_start}^T ||u_t||_2 + ||d_t||_1
	# subject to x_{t+1} = growth*x_t - damage*u_t + w_t.
	obj = pnorm(u[t_s:],2)
	constr = [x[t+1] == growth_mat*x[t] - damage_mat*u[t] for t in range(t_s,T)]
	constr += [x[t_s] == x_cur, x_PTV[(t_s+1):] <= PTV_max[(t_s+1):], x_PTV[(t_s+1):] >= PTV_min,
			   x_OAR[(t_s+1):] >= OAR_min[(t_s+1):], u[t_s:] >= 0, u[T_treat:] == 0]
	
	# Solve problem.
	prob = Problem(Minimize(obj), constr)
	prob.solve(solver = "MOSEK")
	print("Start Time:", t_s)
	print("Status:", prob.status)
	print("Objective:", prob.value)
	
	# Update actual health status and treatment.
	w_true = health_noise((m,), mu, sigma)
	x_cur = x[t_s+1].value + w_true
	x_final[t_s+1] = x_cur
	u_final[t_s] = u[t_s].value
	w_final[t_s] = w_true

# Fill in recovery period.
for t_s in range(T_treat,T):
	w_true = health_noise((m,), mu, sigma)
	x_final[t_s+1] = growth_mat.dot(x_final[t_s]) + w_true
	u_final[t_s] = 0
	w_final[t_s] = w_true
print("\nFinal Objective:", pnorm(u_final,2).value)

# Plot health and treatment curves.
x_prog = health_prognosis(x_init, growth_mat, T, w_noise = w_final)
plot_health(x_final, {"Untreated": x_prog}, stepsize = 10, T_treat = T_treat)
plot_treatment(u_final, stepsize = 10, T_treat = T_treat)
# plot_health(x_final, {"Untreated": x_prog}, stepsize = 10, T_treat = T_treat, filename = "mpc_simple_health.png")
# plot_treatment(u_final, stepsize = 10, T_treat = T_treat, filename = "mpc_simple_treatment.png")
