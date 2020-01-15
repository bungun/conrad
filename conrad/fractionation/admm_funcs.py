import cvxpy
import numpy as np
import numpy.linalg as LA
from cvxpy import *
from time import time
from multiprocessing import Process, Pipe
from data_utils import pad_matrix, health_prognosis
from mpc_funcs import dose_penalty, health_penalty, dyn_objective, rx_to_constrs

def build_dyn_prob_dose(A_list, patient_rx, T_recov = 0):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	
	# Define variables.
	b = Variable((T_treat,n), pos = True, name = "beams")   # Beams.
	d = vstack([A_list[t]*b[t] for t in range(T_treat)])    # Doses.
	
	# Dose penalty function.
	obj = sum([dose_penalty(d[t], patient_rx["dose_weights"]) for t in range(T_treat)])
	
	# Additional dose constraints.
	# constrs = [b >= 0]
	constrs = []
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(vstack(d), patient_rx["dose_constrs"])
	
	prob = Problem(Minimize(obj), constrs)
	return prob, b, d

def build_dyn_prob_dose_period(A, patient_rx, T_recov = 0):
	K, n = A.shape
	
	# Define variables for period.
	b_t = Variable(n, pos = True, name = "beams")   # Beams.
	d_t = A*b_t
	
	# Dose penalty current period.
	obj = dose_penalty(d_t, patient_rx["dose_weights"])
	
	# Additional dose constraints in period.
	# constrs = [b >= 0]
	constrs = []
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(d_t, patient_rx["dose_constrs"])
	
	prob_t = Problem(Minimize(obj), constrs)
	return prob_t, b_t, d_t

def build_dyn_prob_health(F, G, r, h_init, patient_rx, T_treat, T_recov = 0):
	K = h_init.shape[0]
	
	# Define variables.
	h = Variable((T_treat+1,K), name = "health")   # Health statuses.
	d = Variable((T_treat,K), pos = True, name = "doses")   # Doses.
	
	# Health penalty function.
	obj = sum([health_penalty(h[t+1], patient_rx["health_goal"], patient_rx["health_weights"]) for t in range(T_treat)])
	
	# Health dynamics for treatment stage.
	constrs = [h[0] == h_init]
	for t in range(T_treat):
		constrs.append(h[t+1] == F*h[t] + G*d[t] + r)
	
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
	return prob, h, d

def run_dose_worker(pipe, A, patient_rx, T_recov, rho, *args, **kwargs):
	# Construct proximal dose problem.
	prob_dose, b, d = build_dyn_prob_dose_period(A, patient_rx, T_recov)
	d_new = Parameter(d.shape, value = np.zeros(d.shape))
	u = Parameter(d.shape, value = np.zeros(d.shape))
	penalty = (rho/2)*sum_squares(d - d_new - u)
	prox = prob_dose + Problem(Minimize(penalty))
	
	# ADMM loop.
	finished = False
	while not finished:
		# Compute and send d_t^k.
		prox.solve(*args, **kwargs)
		if prox.status not in ["optimal", "optimal_inaccurate"]:
			raise RuntimeError("Solver failed with status {0}".format(prox.status))
		pipe.send((d.value, prox.solver_stats.solve_time))
		
		# Receive \tilde d_t^k.
		d_new.value = pipe.recv()
		
		# Update and send u_t^^k.
		u.value += d_new.value - d.value
		pipe.send(u.value)
		
		# Check if stopped.
		finished = pipe.recv()
	
	# Send final b_t^k and p_t^k.
	pipe.send(b.value)

def dynamic_treatment_admm(A_list, F, G, r, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, partial_results = False, *args, **kwargs):
	# Problem parameters.
	max_iter = kwargs.pop("max_iter", 1000) # Maximum iterations.
	rho = kwargs.pop("rho", 1/10)           # Step size.
	eps_abs = kwargs.pop("eps_abs", 1e-6)   # Absolute stopping tolerance.
	eps_rel = kwargs.pop("eps_rel", 1e-8)   # Relative stopping tolerance.
	verbose = kwargs.get("verbose", False)
	
	# Validate parameters.
	T_treat = len(A_list)
	K = F.shape[1]
	if max_iter <= 0:
		raise ValueError("max_iter must be a positive integer.")
	if rho <= 0:
		raise ValueError("rho must be a positive scalar.")
	if eps_abs < 0:
		raise ValueError("eps_abs must be a non-negative scalar.")
	if eps_rel < 0:
		raise ValueError("eps_rel must be a non-negative scalar.")
		
	# Set up dose workers.
	pipes = []
	procs = []
	for t in range(T_treat):
		rx_cur = patient_rx.copy()
		# rx_cur["health_goal"] = patient_rx["health_goal"][t]
		if "dose_constrs" in patient_rx:
			rx_cur["dose_constrs"] = {"lower": patient_rx["dose_constrs"]["lower"][t], "upper": patient_rx["dose_constrs"]["upper"][t]}
		if "health_constrs" in patient_rx:
			rx_cur["health_constrs"] = {"lower": patient_rx["health_constrs"]["lower"][t], "upper": patient_rx["health_constrs"]["upper"][t]}
			
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target=run_dose_worker, args=(remote, A_list[t], rx_cur, T_recov, rho) + args, kwargs=kwargs)]
		procs[-1].start()
	
	# Proximal health problem.
	prob_health, h, d_tld = build_dyn_prob_health(F, G, r, h_init, patient_rx, T_treat, T_recov)
	d_new = Parameter(d_tld.shape, value = np.zeros(d_tld.shape))
	u = Parameter(d_tld.shape, value = np.zeros(d_tld.shape))
	penalty = (rho/2)*sum_squares(d_tld - d_new + u)
	prox = prob_health + Problem(Minimize(penalty))
	
	# ADMM loop.
	k = 0
	finished = False
	r_prim = np.zeros(max_iter)
	r_dual = np.zeros(max_iter)
	
	start = time()
	solve_time = 0
	while not finished:
		if verbose and k % 10 == 0:
			print("Iteration:", k)
            
		# Collect and stack d_t^k for t = 1,...,T.
		dt_update = [pipe.recv() for pipe in pipes]
		d_rows, d_times = map(list, zip(*dt_update))
		d_new.value = np.row_stack(d_rows)
		solve_time += np.max(d_times)
		
		# Compute and send \tilde d_t^k.
		d_tld_prev = np.zeros((T_treat,K)) if k == 0 else d_tld.value
		prox.solve(*args, **kwargs)
		if prox.status not in ["optimal", "optimal_inaccurate"]:
			raise RuntimeError("Solver failed with status {0}".format(prox.status))
		solve_time += prox.solver_stats.solve_time
		for t in range(T_treat):
			pipes[t].send(d_tld[t].value)
		
		# Receive and update u_t^k for t = 1,...,T.
		u_rows = [pipe.recv() for pipe in pipes]
		u.value = np.row_stack(u_rows)
		
		# Calculate residuals.
		r_prim_mat = d_new.value - d_tld.value
		r_dual_mat = rho*(d_tld.value - d_tld_prev)
		r_prim[k] = LA.norm(r_prim_mat)
		r_dual[k] = LA.norm(r_dual_mat)
		
		# Check stopping criteria.
		eps_prim = eps_abs*np.sqrt(T_treat*K) + eps_rel*np.maximum(LA.norm(d_new.value), LA.norm(d_tld.value))
		eps_dual = eps_abs*np.sqrt(T_treat*K) + eps_rel*LA.norm(u.value)
		finished = (k + 1) >= max_iter or (r_prim[k] <= eps_prim and r_dual[k] <= eps_dual)
		k = k + 1
		for pipe in pipes:
			pipe.send(finished)
	
	# Receive final values of b_t^k for t = 1,...,T.
	b_rows = [pipe.recv() for pipe in pipes]
	b_val = np.row_stack(b_rows)
	
	[proc.terminate() for proc in procs]
	end = time()
	
	# Only used internally for calls in MPC.
	if partial_results:
		# TODO: Return primal/dual residuals as well?
		obj_pred = dyn_objective(d_tld.value, h.value, patient_rx).value
		return {"obj": obj_pred, "status": prox.status, "solve_time": solve_time, "beams": b_val, "doses": d_tld.value}
	
	# Construct full results.
	beams_all = pad_matrix(b_val, T_recov)
	doses_all = pad_matrix(d_tld.value, T_recov)
	# doses_all = pad_matrix((d_tld.value + d_new.value)/2, T_recov)
	health_all = health_prognosis(F, h_init, T_treat + T_recov, G, r, doses_all, health_map)
	obj = dyn_objective(d_tld.value, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj, "status": prox.status, "num_iters": k, "total_time": end - start, "solve_time": solve_time, 
			"beams": beams_all, "doses": doses_all, "health": health_all, "primal": np.array(r_prim[:k]), "dual": np.array(r_dual[:k])}

def mpc_treatment_admm(A_list, F, G, r, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, mpc_verbose = False, *args, **kwargs):
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
		result = dynamic_treatment_admm(A_list[t_s:], F, G, r, h_cur, rx_cur, T_recov, partial_results = True, *args, **kwargs)
		solve_time += result["solve_time"]
		
		if mpc_verbose:
			print("Start Time:", t_s)
			print("Status:", result["status"])
			print("Objective:", result["obj"])
			print("Solve Time:", result["solve_time"])
		
		# Save beam, doses, and penalties for current period.
		status = result["status"]
		beams[t_s] = result["beams"][0]
		doses[t_s] = result["doses"][0]
		
		# Update health for next period.
		h_cur = health_map(F.dot(h_cur) + G.dot(doses[t_s]) + r, t_s)
	
	# Construct full results.
	beams_all = pad_matrix(beams, T_recov)
	doses_all = pad_matrix(doses, T_recov)
	health_all = health_prognosis(F, h_init, T_treat + T_recov, G, r, doses_all, health_map)
	obj_treat = dyn_objective(doses, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj_treat, "status": status, "solve_time": solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}
