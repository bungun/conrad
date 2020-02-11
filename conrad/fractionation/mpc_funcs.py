import cvxpy
import numpy as np
from cvxpy import *
from data_utils import pad_matrix, check_dyn_matrices, health_prognosis

def rx_trunc(patient_rx, t_s):
	rx_cur = patient_rx.copy()
	# rx_cur["health_goal"] = patient_rx["health_goal"][t_s:]
	
	# if "beam_constrs" in patient_rx:
	#	rx_cur["beam_constrs"] = {"lower": patient_rx["beam_constrs"]["lower"][t_s:], "upper": patient_rx["beam_constrs"]["upper"][t_s:]}
	# if "dose_constrs" in patient_rx:
	#	rx_cur["dose_constrs"] = {"lower": patient_rx["dose_constrs"]["lower"][t_s:], "upper": patient_rx["dose_constrs"]["upper"][t_s:]}
	# if "health_constrs" in patient_rx:
	#	rx_cur["health_constrs"] = {"lower": patient_rx["health_constrs"]["lower"][t_s:], "upper": patient_rx["health_constrs"]["upper"][t_s:]}
	
	for constr_key in {"beam_constrs", "dose_constrs", "health_constrs"}:
		if constr_key in patient_rx:
			rx_cur[constr_key] = {}
			for lu_key in {"lower", "upper"}:
				if lu_key in patient_rx[constr_key]:
					rx_cur[constr_key][lu_key] = patient_rx[constr_key][lu_key][t_s:]
	return rx_cur

# Dose penalty per period.
def dose_penalty(dose, goal = None, weights = None):
	if goal is None:
		goal = np.zeros(dose.shape)
	if weights is None:
		weights = np.ones(dose.shape)
	return weights*square(dose - goal)

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
		d_penalty = dose_penalty(d_var[t], patient_rx["dose_goal"], patient_rx["dose_weights"])
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
def build_dyn_prob(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov = 0):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	if h_init.shape[0] != K:
		raise ValueError("h_init must be a vector of {0} elements". format(K))
	
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
		constrs.append(h[t+1] == F_list[t]*h[t] + G_list[t]*d[t] + r_list[t])
	
	# Additional beam constraints.
	if "beam_constrs" in patient_rx:
		constrs += rx_to_constrs(b, patient_rx["beam_constrs"])
	
	# Additional dose constraints.
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(d, patient_rx["dose_constrs"])
	
	# Additional health constraints.
	if "health_constrs" in patient_rx:
		constrs += rx_to_constrs(h[1:], patient_rx["health_constrs"])
	
	# Health dynamics for recovery stage.
	# TODO: Should we return h_r or calculate it later?
	if T_recov > 0:
		F_recov = F_list[T_treat:]
		r_recov = r_list[T_treat:]
		
		h_r = Variable((T_recov,K), name = "recovery")
		constrs_r = [h_r[0] == F_recov[0]*h[-1] + r_recov[0]]
		for t in range(T_recov-1):
			constrs_r.append(h_r[t+1] == F_recov[t+1]*h_r[t] + r_recov[t+1])
		
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
	
	obj = dose_penalty(d, patient_rx["dose_goal"], patient_rx["dose_weights"])
	# constrs = [d == A*b, b >= 0]
	constrs = [d == A*b]
	
	if "beam_constrs" in patient_rx:
		constrs += rx_to_constrs(b, patient_rx["beam_constrs"])
	
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(d, patient_rx["dose_constrs"])
	
	prob = Problem(Minimize(obj), constrs)
	prob.solve(*args, **kwargs)
	# h = F.dot(h_init) + G.dot(d.value)
	return {"obj": prob.value, "status": prob.status, "beams": b.value, "doses": d.value}

def dynamic_treatment(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	F_list, G_list, r_list = check_dyn_matrices(F_list, G_list, r_list, K, T_treat, T_recov)
	
	# Build problem for treatment stage.
	prob, b, h, d = build_dyn_prob(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov)
	prob.solve(*args, **kwargs)
	if prob.status not in ["optimal", "optimal_inaccurate"]:
		raise RuntimeError("Solver failed with status {0}".format(prob.status))
	
	# Construct full results.
	beams_all = pad_matrix(b.value, T_recov)
	doses_all = pad_matrix(d.value, T_recov)
	G_list_pad = G_list + T_recov*[np.zeros(G_list[0].shape)]
	health_all = health_prognosis(h_init, T_treat + T_recov, F_list, G_list_pad, r_list, doses_all, health_map)
	obj = dyn_objective(d.value, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj, "status": prob.status, "solve_time": prob.solver_stats.solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}

def mpc_treatment(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, mpc_verbose = False, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	F_list, G_list, r_list = check_dyn_matrices(F_list, G_list, r_list, K, T_treat, T_recov)
	
	# Initialize values.
	beams = np.zeros((T_treat,n))
	doses = np.zeros((T_treat,K))
	solve_time = 0
	
	h_cur = h_init
	for t_s in range(T_treat):
		# Drop prescription for previous periods.
		rx_cur = rx_trunc(patient_rx, t_s)
		
		# Solve optimal control problem from current period forward.
		T_left = T_treat - t_s
		prob, b, h, d = build_dyn_prob(T_left*[A_list[t_s]], T_left*[F_list[t_s]], T_left*[G_list[t_s]], T_left*[r_list[t_s]], h_cur, rx_cur, T_recov)
		# prob, b, h, d = build_dyn_prob(A_list[t_s:], F_list[t_s:], G_list[t_s:], r_list[t_s:], h_cur, rx_cur, T_recov)
		prob.solve(*args, **kwargs)
		if prob.status not in ["optimal", "optimal_inaccurate"]:
			raise RuntimeError("Solver failed with status {0}".format(prob.status))
		solve_time += prob.solver_stats.solve_time
		
		if mpc_verbose:
			print("Start Time:", t_s)
			print("Status:", prob.status)
			print("Objective:", prob.value)
			print("Solve Time:", prob.solver_stats.solve_time)
		
		# Save beams, doses, and penalties for current period.
		status = prob.status
		beams[t_s] = b.value[0]
		doses[t_s] = d.value[0]
		
		# Update health for next period.
		h_cur = health_map(h.value[1], t_s)
		# h_cur = health_map(F_list[t_s].dot(h_cur) + G_list[t_s].dot(doses[t_s]) + r_list[t_s], t_s)
	
	# Construct full results.
	beams_all = pad_matrix(beams, T_recov)
	doses_all = pad_matrix(doses, T_recov)
	G_list_pad = G_list + T_recov*[np.zeros(G_list[0].shape)]
	health_all = health_prognosis(h_init, T_treat + T_recov, F_list, G_list_pad, r_list, doses_all, health_map)
	obj_treat = dyn_objective(doses, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj_treat, "status": status, "solve_time": solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}
