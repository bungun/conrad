import cvxpy
import numpy as np
from cvxpy import *
from data_utils import pad_matrix, health_prognosis

# Dose penalty per period.
def dose_penalty(dose, weights):
	return weights*square(dose)

# Health status penalty per period.
def health_penalty(health, goal, weights):
	w_under, w_over = weights
	return w_under*neg(health - goal) + w_over*pos(health - goal)

# Full objective function.
def dyn_objective(d_var, h_var, patient_rx):
	T, K = d_var.shape
	if h_var.shape[0] != T+1:
		raise ValueError("h_var must have exactly {0} rows".format(T+1))
	
	penalties = []
	for t in range(T):
		d_penalty = dose_penalty(d_var[t], patient_rx["dose_weights"])
		h_penalty = health_penalty(h_var[t+1], patient_rx["health_goal"], patient_rx["health_weights"])
		penalties.append(d_penalty + h_penalty)
	return sum(penalties)

# Extract constraints from patient prescription.
def rx_to_constrs(expr, rx_dict):
	constrs = []
	
	# Lower bound.
	if "lower" in rx_dict:
		rx_lower = rx_dict["lower"]
		if np.any(rx_lower == np.inf):
			raise ValueError("Lower bound cannot be infinity")
		
		if np.isscalar(rx_lower):
			if np.isfinite(rx_lower):
				constrs.append(expr >= rx_lower)
		else:
			if rx_lower.shape != expr.shape:
				raise ValueError("rx_lower must have dimensions {0}".format(expr.shape))
			is_finite = np.isfinite(rx_lower)
			if np.any(is_finite):
				constrs.append(expr[is_finite] >= rx_lower[is_finite])
		
	# Upper bound.
	if "upper" in rx_dict:
		rx_upper = rx_dict["upper"]
		if np.any(rx_upper == -np.inf):
			raise ValueError("Upper bound cannot be negative infinity")
		
		if np.isscalar(rx_upper):
			if np.isfinite(rx_upper):
				constrs.append(expr <= rx_upper)
		else:
			if rx_upper.shape != expr.shape:
				raise ValueError("rx_upper must have dimensions {0}".format(expr.shape))
			is_finite = np.isfinite(rx_upper)
			if np.any(is_finite):
				constrs.append(expr[is_finite] <= rx_upper[is_finite])
	return constrs

# Construct optimal control problem.
def build_dyn_prob(A_list, F, G, r, h_init, patient_rx, T_recov = 0):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	
	if h_init.shape[0] != K:
		raise ValueError("h_init must be a vector of {0} elements". format(K))
	if F.shape != (K,K):
		raise ValueError("F must have dimensions ({0},{0})".format(K))
	if G.shape != (K,K):
		raise ValueError("G must have dimensions ({0},{0})".format(K))
	if r.shape != (K,) and r.shape != (K,1):
		raise ValueError("r must have dimensions ({K},)".format(K))
	
	# Define variables.
	b = Variable((T_treat,n), pos = True, name = "beams")   # Beams.
	h = Variable((T_treat+1,K), name = "health")            # Health statuses.
	d = vstack([A_list[t]*b[t] for t in range(T_treat)])    # Doses.
	
	# Objective function.
	obj = dyn_objective(d, h, patient_rx)
	
	# Health dynamics for treatment stage.
	# constrs = [h[0] == h_init, b >= 0]
	constrs = [h[0] == h_init]
	for t in range(T_treat):
		constrs.append(h[t+1] == F*h[t] + G*d[t] + r)
	
	# Additional dose constraints.
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(vstack(d), patient_rx["dose_constrs"])
	
	# Additional health constraints.
	if "health_constrs" in patient_rx:
		constrs += rx_to_constrs(h[1:], patient_rx["health_constrs"])
	
	# Health dynamics for recovery stage.
	# TODO: Should we return h_r or calculate it later?
	if T_recov > 0:
		h_r = Variable((T_recov,K), name = "recovery")
		constrs_r = [h_r[0] == F*h[-1]]
		for t in range(T_recov-1):
			constrs_r.append(h_r[t+1] == F*h_r[t] + r)
		
		# Additional health constraints during recovery.
		if "recov_constrs" in patient_rx:
			constrs_r += rx_to_constrs(h_r, patient_rx["recov_constrs"])
		constrs += constrs_r
	
	prob = Problem(Minimize(obj), constrs)
	return prob, b, h, d

def single_treatment(A, patient_rx, *args, **kwargs):
	K, n = A.shape
	b = Variable(n, pos = True)   # Beams.
	d = Variable(K, pos = True)   # Doses.
	
	obj = dose_penalty(d, patient_rx["dose_weights"])
	# constrs = [d == A*b, b >= 0]
	constrs = [d == A*b]
	
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(d, patient_rx["dose_constrs"])
	
	prob = Problem(Minimize(obj), constrs)
	prob.solve(*args, **kwargs)
	# h = F.dot(h_init) + G.dot(d.value)
	return {"obj": prob.value, "status": prob.status, "beams": b.value, "doses": d.value}

def dynamic_treatment(A_list, F, G, r, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, *args, **kwargs):
	T_treat = len(A_list)
	
	# Build problem for treatment stage.
	prob, b, h, d = build_dyn_prob(A_list, F, G, r, h_init, patient_rx, T_recov)
	prob.solve(*args, **kwargs)
	
	# Construct full results.
	beams_all = pad_matrix(b.value, T_recov)
	doses_all = pad_matrix(d.value, T_recov)
	health_all = health_prognosis(F, h_init, T_treat + T_recov, G, r, doses_all, health_map)
	obj = dyn_objective(d.value, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj, "status": prob.status, "solve_time": prob.solver_stats.solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}

def mpc_treatment(A_list, F, G, r, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, mpc_verbose = False, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	
	# Initialize values.
	beams = np.zeros((T_treat,n))
	doses = np.zeros((T_treat,K))
	solve_time = 0
	
	h_cur = h_init
	for t_s in range(T_treat):
		rx_cur = patient_rx.copy()
		# rx_cur["health_goal"] = patient_rx["health_goal"][t_s:]
		if "dose_constrs" in patient_rx:
			rx_cur["dose_constrs"] = {"lower": patient_rx["dose_constrs"]["lower"][t_s:], "upper": patient_rx["dose_constrs"]["upper"][t_s:]}
		if "health_constrs" in patient_rx:
			rx_cur["health_constrs"] = {"lower": patient_rx["health_constrs"]["lower"][t_s:], "upper": patient_rx["health_constrs"]["upper"][t_s:]}
		
		# Solve optimal control problem from current period forward.
		prob, b, h, d = build_dyn_prob(A_list[t_s:], F, G, r, h_cur, rx_cur, T_recov)
		prob.solve(*args, **kwargs)
		if prob.status not in ["optimal", "optimal_inaccurate"]:
			raise RuntimeError("Solver failed with status {0}".format(prob.status))
		solve_time += prob.solver_stats.solve_time
		
		if mpc_verbose:
			print("Start Time:", t_s)
			print("Status:", prob.status)
			print("Objective:", prob.value)
			print("Solve Time:", prob.solve_time)
		
		# Save beam, doses, and penalties for current period.
		status = prob.status
		beams[t_s] = b.value[0]
		doses[t_s] = d.value[0]
		
		# Update health for next period.
		h_cur = health_map(F.dot(h_cur) + G.dot(doses[t_s]) + r, t_s)
	
	# Construct full results.
	beams_all = pad_matrix(beams, T_recov)
	doses_all = pad_matrix(doses, T_recov)
	health_all = health_prognosis(F, h_init, T_treat + T_recov, G, r, doses_all, health_map)
	obj_treat = dyn_objective(doses, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj_treat, "status": status, "solve_time": solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}
