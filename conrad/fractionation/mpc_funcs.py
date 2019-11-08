import cvxpy
import numpy as np
from cvxpy import *
from data_utils import pad_matrix

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

def dose_penalty(dose, prescription, weights):
	w_under, w_over = weights
	return w_under*neg(dose - prescription) + w_over*pos(dose - prescription)

def rx_to_constrs(expr, rx_dict):
	constrs = []
	T, K = expr.shape
	
	# Lower bound.
	if "lower" in rx_dict:
		rx_lower = rx_dict["lower"]
		if np.any(rx_lower == np.inf):
			raise ValueError("Lower bound cannot be infinity")
		
		if np.isscalar(rx_lower):
			if np.isfinite(rx_lower):
				constrs.append(expr >= rx_lower)
		else:
			if rx_lower.shape != (T,K):
				raise ValueError("rx_lower must have dimensions ({0},{1})".format(T,K))
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
			if rx_upper.shape != (T,K):
				raise ValueError("rx_upper must have dimensions ({0},{1})".format(T,K))
			is_finite = np.isfinite(rx_upper)
			if np.any(is_finite):
				constrs.append(expr[is_finite] <= rx_upper[is_finite])
	return constrs

def build_dyn_prob(A_list, F, G, h_init, patient_rx, T_recov = 0):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	rx_dose = patient_rx["dose"]
	rx_weights = patient_rx["weights"]
	
	if h_init.shape[0] != K:
		raise ValueError("h_init must be a vector of {0} elements". format(K))
	if F.shape != (K,K):
		raise ValueError("F must have dimensions ({0},{0})".format(K))
	if G.shape != (K,K):
		raise ValueError("G must have dimensions ({0},{0})".format(K))
	
	# Define variables.
	b = Variable((T_treat,n), pos = True, name = "beams")   # Beams.
	h = Variable((T_treat+1,K), name = "health")              # Health statuses.
	d = vstack([A_list[t]*b[t] for t in range(T_treat)])      # Doses.
	
	# Dose penalties and constraints.
	penalties = []
	# constrs = [h[0] == h_init, b >= 0]
	constrs = [h[0] == h_init]
	for t in range(T_treat):
		penalty = dose_penalty(d[t], rx_dose, rx_weights)
		penalties.append(penalty)
		constrs.append(h[t+1] == F*h[t] + G*d[t])
	
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
			constrs_r.append(h_r[t+1] == F*h_r[t])
		
		# Additional health constraints during recovery.
		if "recov_constrs" in patient_rx:
			constrs_r += rx_to_constrs(h_r, patient_rx["recov_constrs"])
		constrs += constrs_r
	
	obj = sum(penalties)
	prob = Problem(Minimize(obj), constrs)
	return prob, b, h, d

def single_treatment(A, patient_rx, *args, **kwargs):
	K, n = A.shape
	rx_dose = patient_rx["dose"]
	rx_weights = patient_rx["weights"]
	
	b = Variable(n, pos = True)   # Beams.
	# d = Variable(K, pos = True)
	d = Variable(K, pos = True)   # Doses.
	
	obj = dose_penalty(d, rx_dose, rx_weights)
	# constrs = [d == A*b, b >= 0]
	constrs = [d == A*b]
	
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(d, patient_rx["dose_constrs"])
	
	prob = Problem(Minimize(obj), constrs)
	prob.solve(*args, **kwargs)
	# h = F.dot(h_init) + G.dot(d.value)
	return {"obj": prob.value, "status": prob.status, "beams": b.value, "doses": d.value}

def dynamic_treatment(A_list, F, G, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, *args, **kwargs):
	T_treat = len(A_list)
	
	# Build problem for treatment stage.
	prob, b, h, d = build_dyn_prob(A_list, F, G, h_init, patient_rx, T_recov)
	prob.solve(*args, **kwargs)
	
	# Construct full results.
	beams_all = pad_matrix(b.value, T_recov)
	doses_all = pad_matrix(d.value, T_recov)
	health_all = health_prognosis_gen(F, h_init, T_treat + T_recov, G, doses_all, health_map)
	return {"obj": prob.value, "status": prob.status, "solve_time": prob.solver_stats.solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}

def mpc_treatment(A_list, F, G, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, mpc_verbose = False, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	rx_dose = patient_rx["dose"]
	rx_weights = patient_rx["weights"]
	
	# Initialize values.
	beams = np.zeros((T_treat,n))
	doses = np.zeros((T_treat,K))
	penalties = np.zeros(T_treat)
	solve_time = 0
	
	h_cur = h_init
	for t_s in range(T_treat):
		rx_cur = patient_rx.copy()
		if "dose_constrs" in patient_rx:
			rx_cur["dose_constrs"] = {"lower": patient_rx["dose_constrs"]["lower"][t_s:], "upper": patient_rx["dose_constrs"]["upper"][t_s:]}
		if "health_constrs" in patient_rx:
			rx_cur["health_constrs"] = {"lower": patient_rx["health_constrs"]["lower"][t_s:], "upper": patient_rx["health_constrs"]["upper"][t_s:]}
		
		# Solve optimal control problem from current period forward.
		prob, b, h, d = build_dyn_prob(A_list[t_s:], F, G, h_cur, rx_cur, T_recov)
		prob.solve(*args, **kwargs)
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
		penalties[t_s] = dose_penalty(doses[t_s], rx_dose, rx_weights).value
		
		# Update health for next period.
		h_cur = health_map(F.dot(h_cur) + G.dot(doses[t_s]), t_s)
	
	# Construct full results.
	beams_all = pad_matrix(beams, T_recov)
	doses_all = pad_matrix(doses, T_recov)
	health_all = health_prognosis_gen(F, h_init, T_treat + T_recov, G, doses_all, health_map)
	return {"obj": np.sum(penalties), "status": status, "solve_time": solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}
