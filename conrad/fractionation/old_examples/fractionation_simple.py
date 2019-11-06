import numpy as np
import cvxpy
from cvxpy import *
from data_utils import *
from plot_utils import *

# Growth rates.          x_0   x_1   x_2   x_3
growth_mat = np.array([[1.02,    0,    0,    0],   # x_0 = PTV
					   [   0, 0.95,    0,    0],   # x_1 = OAR_1
					   [   0,    0, 0.90,    0],   # x_2 = OAR_2
					   [   0,    0,    0, 0.75]])  # x_3 = OAR_3

# growth_mat = np.array([[  1.02,    0,    0,    0],   # x_0 = PTV
#  				     [-0.075, 0.95,    0,    0],   # x_1 = OAR_1
#  				     [ -0.05,    0, 0.90,    0],   # x_2 = OAR_2
#  				     [ -0.02,    0,    0, 0.75]])  # x_3 = OAR_3
					   
# Damage rates.		     u_0   u_1    u_2
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
# OAR_min = np.full((T+1,m-1), -0.05)
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

# Minimize \sum_{t=1}^T ||u_t||_2
# subject to x_{t+1} = growth*x_t - damage*u_t.
obj = pnorm(u,2)
constr = [x[t+1] == growth_mat*x[t] - damage_mat*u[t] for t in range(T)]
constr += [x[0] == x_init, x_PTV <= PTV_max, x_PTV >= PTV_min, x_OAR >= OAR_min, 
		   u >= 0, u[T_treat:] == 0]

# Solve problem.
prob = Problem(Minimize(obj), constr)
prob.solve()
print("Status:", prob.status)
print("Objective:", prob.value)

# Calculate health status with individual plans.
curves = {}
# for j in range(n-1):
#	label = "Plan {0}".format(j)
#	curves[label] = health_invariant(x_init, growth_mat, T, damage_mat[:,j])
x_prog = health_prognosis(x_init, growth_mat, T)
curves["Untreated"] = x_prog

# Plot health and treatment curves.
plot_health(x.value, curves = curves, stepsize = 10, T_treat = T_treat)
plot_treatment(u.value, stepsize = 10, T_treat = T_treat)
# plot_health(x.value, curves = curves, stepsize = 10, T_treat = T_treat, filename = "simple_health.png")
# plot_treatment(u.value, stepsize = 10, T_treat = T_treat, filename = "simple_treatment.png")
