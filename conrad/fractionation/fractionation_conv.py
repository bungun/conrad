import numpy as np
import cvxpy
from cvxpy import *
from data_utils import *
from plot_utils import *

m = 4   # Number of health measures.
n = 3   # Number of treatment plans.
T_treat = 14   # Treatment period.
T_recov = 20   # Recovery period.
T = T_treat + T_recov  # Number of time periods.

# Initial health status.
OAR_start = 0
PTV_start = 0.25
x_init = np.array([PTV_start] + (m-1)*[OAR_start])
x_final = np.array([[0, 0, PTV_start],
				    [0, 0, OAR_start],
				    [0, 0, OAR_start],
				    [0, 0, OAR_start]])

# Impulse response matrices.
peak_mat = np.array([[(4,  0   ), (8,  0   ), (4, x_init[0] + 0.05)],
				     [(4, -0.05), (8, -0.1 ), (4, x_init[1] + 0.05)],
				     [(4, -0.1 ), (8, -0.15), (4, x_init[2] + 0.05)],
				     [(4, -0.15), (8, -0.2 ), (4, x_init[3] + 0.05)]])

half_mat = np.array([[6, 10, 6],
					 [6, 10, 6],
					 [6, 10, 6],
					 [6, 10, 6]])

# Impulse response functions.
stepsize = 0.01
T_dim = int(T/stepsize)
H = np.zeros((T_dim,m,n))
for i in range(m):
	for j in range(n):
		t_range, H[:,i,j] = second_order_curve(peak_mat[i,j], x_init[i], x_final[i,j], half_mat[i,j], T = T, stepsize = stepsize)
plot_impulse(H, t_range, stepsize = 10, T_treat = T_treat, filename = "impulse_response.png")
	
# Constraints.
OAR_min = np.full((T+1,m-1), -0.25)
PTV_worst = 0.5
PTV_end = 0.001
PTV_max = np.array([PTV_start] + T_treat*[PTV_worst] + T_recov*[PTV_end])

# Construct problem.
x = Variable((T+1,m))   # Health (state).
u = Variable((T+1,n))   # Treatment (control).

# Split into PTVs and OARs.
x_PTV = x[:,0]
x_OAR = x[:,1:]

# Minimize \sum_{t=1}^T ||u_t||_2
# subject to x_t = (H * u)(t) = \sum_{\tau} H(t-\tau)u(\tau).
obj = pnorm(u,2)
constr = []
for i in range(m):
	expr = sum([conv(H[:,i,j], u[:,j]) for j in range(n)])
	constr += [x[t,i] == expr[t,0] for t in range(T)]
constr += [x_PTV[T] == PTV_end, x_PTV <= PTV_max, x_OAR >= OAR_min, 
		   u >= 0, u <= 1, u[T_treat:] == 0]
	
# Solve problem.
prob = Problem(Minimize(obj), constr)
prob.solve()
print("Status:", prob.status)
print("Objective:", prob.value)

# Plot health and treatment curves.
plot_health(x.value, stepsize = 10, T_treat = T_treat)
plot_treatment(u.value, stepsize = 10, T_treat = T_treat)
