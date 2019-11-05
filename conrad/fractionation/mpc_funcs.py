import cvxpy
import numpy as np
from cvxpy import *
from data_utils import pad_matrix

def dose_penalty(dose, prescription, weights):
	w_under, w_over = weights
	return w_under*neg(dose - prescription) + w_over*pos(dose - prescription)

def health_prognosis_gen(F, h_init, T, G = None, doses = None, health_map = lambda h,t: h):
	K = h_init.shape[0]
	h_prog = np.zeros((T+1,K))
	h_prog[0] = h_init
	
	# Defaults to no treatment.
	if G is None and doses is None:
		G = np.zeros((K,1))
		doses = np.zeros((T,1))
	elif not (doses is not None and G is not None):
		raise ValueError("Both G and doses must be provided.")
	
	for t in range(T):
		h_prog[t+1] = health_map(F.dot(h_prog[t]) + G.dot(doses[t]), t)
	return h_prog

def recovery_stage(F, h_final, T_recov, health_map = lambda h,t: h):
	return health_prognosis_gen(F, h_final, T_recov, health_map = health_map)[1:]

# TODO: Allow user to add optional constraints on beams/doses.
def single_treatment(A, rx_dose, rx_weights, *args, **kwargs):
	K, n = A.shape
	b = Variable(n, pos = True)   # Beams.
	# d = Variable(K, pos = True)
	d = Variable(K, pos = True)   # Doses.
	
	obj = dose_penalty(d, rx_dose, rx_weights)
	# constrs = [d == A*b, b >= 0]
	constrs = [d == A*b]
	prob = Problem(Minimize(obj), constrs)
	prob.solve(*args, **kwargs)
	# h = F.dot(h_init) + G.dot(d.value)
	return {"obj": prob.value, "status": prob.status, "beams": b.value, "doses": d.value}

def dynamic_treatment(A_list, F, G, h_init, rx_dose, rx_weights, health_map = lambda h,t: h, *args, **kwargs):
	T = len(A_list)
	K, n = A_list[0].shape
	if F.shape != (K,K):
		raise ValueError("F must have dimensions ({0},{0})".format(K))
	if G.shape != (K,K):
		raise ValueError("G must have dimensions ({0},{0})".format(K))
	
	# Define variables.
	b = Variable((T+1,n), pos = True)   # Beams.
	h = Variable((T+1,K))   # Health statuses.
	
	# Dose penalties and constraints.
	d = []
	penalties = []
	# constrs = [h[0] == h_init, b >= 0]
	constrs = [h[0] == h_init]
	for t in range(T):
		d.append(A_list[t]*b[t])   # Doses.
		penalty = dose_penalty(d[t], rx_dose, rx_weights)
		penalties.append(penalty)
		constrs.append(h[t+1] == F*h[t] + G*d[t])
	
	# Solve problem.
	obj = sum(penalties)
	prob = Problem(Minimize(obj), constrs)
	prob.solve(*args, **kwargs)
	doses = np.row_stack([dose.value for dose in d])
	
	# Calculate true health status with treatment.
	h_actual = np.zeros((T+1,K))
	h_actual[0] = h_init
	for t in range(T):
		h_actual[t+1] = health_map(F.dot(h_actual[t]) + G.dot(doses[t]), t)
	return {"obj": prob.value, "status": prob.status, "beams": b.value, "doses": doses, "health": h_actual}

def dynamic_treat_recover(A_list, F, G, h_init, rx_dose, rx_weights, T_recov = 0, health_map = lambda h,t: h, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	
	# Treatment stage.
	result_treat = dynamic_treatment(A_list, F, G, h_init, rx_dose, rx_weights, health_map, *args, **kwargs)
	health_treat = result_treat["health"]

	# Recovery stage.
	health_recov = recovery_stage(F, health_treat[-1], T_recov, lambda h,t: health_map(h, t + T_treat))
	
	# Construct full results.
	beams_all = pad_matrix(result_treat["beams"], T_recov)
	doses_all = pad_matrix(result_treat["doses"], T_recov)
	health_all = np.row_stack([health_treat, health_recov])
	return {"obj": result_treat["obj"], "status": result_treat["status"], "beams": beams_all, "doses": doses_all, "health": health_all}

def mpc_treatment(A_list, F, G, h_init, rx_dose, rx_weights, health_map = lambda h,t: h, mpc_verbose = False, *args, **kwargs):
	T = len(A_list)
	K, n = A_list[0].shape
	if F.shape != (K,K):
		raise ValueError("F must have dimensions ({0},{0})".format(K))
	if G.shape != (K,K):
		raise ValueError("G must have dimensions ({0},{0})".format(K))
	
	# Initialize values.
	beams = np.zeros((T,n))
	doses = np.zeros((T,K))
	health = np.zeros((T+1,K))
	penalties = np.zeros(T)
	
	health[0] = h_init
	for t_s in range(T):
		# Solve optimal control problem from current period forward.
		result = dynamic_treatment(A_list[t_s:], F, G, health[t_s], rx_dose, rx_weights, lambda h,t: health_map(h, t + t_s), *args, **kwargs)
		status = result["status"]
		
		# Print out solution status.
		if mpc_verbose:
			print("Start Time:", t_s)
			print("Status:", result["status"])
			print("Objective:", result["obj"])
		
		# Save beam and doses for current period.
		beams[t_s] = result["beams"][0]
		doses[t_s] = result["doses"][0]
		penalties[t_s] = dose_penalty(doses[t_s], rx_dose, rx_weights).value
		
		# Update health for next period.
		health[t_s+1] = result["health"][1]
		# health[t_s+1] = health_map(F.dot(health[t_s]) + G.dot(doses[t_s]), t_s)
	return {"obj": np.sum(penalties), "status": status, "beams": beams, "doses": doses, "health": health}

def mpc_treat_recover(A_list, F, G, h_init, rx_dose, rx_weights, T_recov = 0, health_map = lambda h,t: h, mpc_verbose = False, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	
	# Treatment stage.
	result_treat = mpc_treatment(A_list, F, G, h_init, rx_dose, rx_weights, health_map, mpc_verbose, *args, **kwargs)
	health_treat = result_treat["health"]

	# Recovery stage.
	health_recov = recovery_stage(F, health_treat[-1], T_recov, lambda h,t: health_map(h, t + T_treat))
	
	# Construct full results.
	beams_all = pad_matrix(result_treat["beams"], T_recov)
	doses_all = pad_matrix(result_treat["doses"], T_recov)
	health_all = np.row_stack([health_treat, health_recov])
	return {"obj": result_treat["obj"], "status": result_treat["status"], "beams": beams_all, "doses": doses_all, "health": health_all}
