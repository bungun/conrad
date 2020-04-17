import cvxpy
import numpy as np
import numpy.linalg as LA
from cvxpy import *
from time import time
from multiprocessing import Process, Pipe
from data_utils import pad_matrix, check_dyn_matrices, health_prognosis
from mpc_funcs import dose_penalty, health_penalty, dyn_objective, rx_to_constrs, rx_slice

def build_dyn_prob_dose(A_list, patient_rx, T_recov = 0):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	if patient_rx["dose_goal"].shape != (T_treat,K):
		raise ValueError("dose_goal must have dimensions ({0},{1})".format(T_treat,K))
	
	# Define variables.
	b = Variable((T_treat,n), pos = True, name = "beams")   # Beams.
	d = vstack([A_list[t]*b[t] for t in range(T_treat)])    # Doses.
	
	# Dose penalty function.
	obj = sum([dose_penalty(d[t], patient_rx["dose_goal"][t], patient_rx["dose_weights"]) for t in range(T_treat)])
	
	# Additional beam constraints.
	# constrs = [b >= 0]
	constrs = []
	if "beam_constrs" in patient_rx:
		constrs += rx_to_constrs(b, patient_rx["beam_constrs"])
	
	# Additional dose constraints.
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
	obj = dose_penalty(d_t, patient_rx["dose_goal"], patient_rx["dose_weights"])
	
	# Additional beam constraints in period.
	# constrs = [b >= 0]
	constrs = []
	if "beam_constrs" in patient_rx:
		constrs += rx_to_constrs(b_t, patient_rx["beam_constrs"])
	
	# Additional dose constraints in period.
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(d_t, patient_rx["dose_constrs"])
	
	prob_t = Problem(Minimize(obj), constrs)
	return prob_t, b_t, d_t

def build_dyn_prob_health(F_list, G_list, r_list, h_init, patient_rx, T_treat, T_recov = 0):
	K = h_init.shape[0]
	if patient_rx["health_goal"].shape != (T_treat,K):
		raise ValueError("health_goal must have dimensions ({0},{1})".format(T_treat,K))
	
	# Define variables.
	h = Variable((T_treat+1,K), name = "health")   # Health statuses.
	d = Variable((T_treat,K), pos = True, name = "doses")   # Doses.
	
	# Health penalty function.
	obj = sum([health_penalty(h[t+1], patient_rx["health_goal"][t], patient_rx["health_weights"]) for t in range(T_treat)])
	
	# Health dynamics for treatment stage.
	constrs = [h[0] == h_init]
	for t in range(T_treat):
		constrs.append(h[t+1] == F_list[t]*h[t] + G_list[t]*d[t] + r_list[t])
	
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
	
	# Send final b_t^k and d_t^k.
	d_val = A.dot(b.value)
	pipe.send((b.value, d_val))

def dynamic_treatment_admm(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, partial_results = False, admm_verbose = False, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	F_list, G_list, r_list = check_dyn_matrices(F_list, G_list, r_list, K, T_treat, T_recov)
	
	# Problem parameters.
	max_iter = kwargs.pop("max_iter", 1000) # Maximum iterations.
	rho = kwargs.pop("rho", 1/10)           # Step size.
	eps_abs = kwargs.pop("eps_abs", 1e-6)   # Absolute stopping tolerance.
	eps_rel = kwargs.pop("eps_rel", 1e-3)   # Relative stopping tolerance.
	
	# Validate parameters.
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
		rx_cur = rx_slice(patient_rx, t, t+1)   # Get prescription at time t.
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target=run_dose_worker, args=(remote, A_list[t], rx_cur, T_recov, rho) + args, kwargs=kwargs)]
		procs[-1].start()
	
	# Proximal health problem.
	prob_health, h, d_tld = build_dyn_prob_health(F_list, G_list, r_list, h_init, patient_rx, T_treat, T_recov)
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
		if admm_verbose and k % 10 == 0:
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
	
	# Receive final values of b_t^k and d_t^k = A*b_t^k for t = 1,...,T.
	bd_final = [pipe.recv() for pipe in pipes]
	b_rows, d_rows = map(list, zip(*bd_final))
	b_val = np.row_stack(b_rows)
	d_val = np.row_stack(d_rows)
	
	[proc.terminate() for proc in procs]
	end = time()
	
	# Only used internally for calls in MPC.
	if partial_results:
		# TODO: Return primal/dual residuals as well?
		# obj_pred = dyn_objective(d_val, h.value, patient_rx).value
		h_val = health_prognosis(h_init, T_treat, F_list[:T_treat], G_list, r_list[:T_treat], d_val, health_map)
		obj_pred = dyn_objective(d_val, h_val, patient_rx).value
		# TODO: Return "weakest" status over all iterations?
		return {"obj": obj_pred, "status": prox.status, "num_iters": k, "solve_time": solve_time, "beams": b_val, "doses": d_val}
	
	# Construct full results.
	beams_all = pad_matrix(b_val, T_recov)
	doses_all = pad_matrix(d_val, T_recov)
	G_list_pad = G_list + T_recov*[np.zeros(G_list[0].shape)]
	health_all = health_prognosis(h_init, T_treat + T_recov, F_list, G_list_pad, r_list, doses_all, health_map)
	obj = dyn_objective(d_val, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj, "status": prox.status, "num_iters": k, "total_time": end - start, "solve_time": solve_time, 
			"beams": beams_all, "doses": doses_all, "health": health_all, "primal": np.array(r_prim[:k]), "dual": np.array(r_dual[:k])}

def mpc_treatment_admm(A_list, F_list, G_list, r_list, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, mpc_verbose = False, *args, **kwargs):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	F_list, G_list, r_list = check_dyn_matrices(F_list, G_list, r_list, K, T_treat, T_recov)
	
	# Initialize values.
	beams = np.zeros((T_treat,n))
	doses = np.zeros((T_treat,K))
	solve_time = 0
	num_iters = 0
	
	h_cur = h_init
	for t_s in range(T_treat):
		# Drop prescription for previous periods.
		rx_cur = rx_slice(patient_rx, t_s, T_treat, squeeze = False)
		
		# Solve optimal control problem from current period forward.
		# TODO: Warm start next ADMM solve, or better yet, rewrite code so no teardown/rebuild process between ADMM solves.
		T_left = T_treat - t_s
		result = dynamic_treatment_admm(T_left*[A_list[t_s]], T_left*[F_list[t_s]], T_left*[G_list[t_s]], T_left*[r_list[t_s]], h_cur, rx_cur, T_recov, partial_results = True, *args, **kwargs)
		# result = dynamic_treatment_admm(A_list[t_s:], F_list[t_s:], G_list[t_s:], r_list[t_s:], h_cur, rx_cur, T_recov, partial_results = True, *args, **kwargs)
		solve_time += result["solve_time"]
		num_iters += result["num_iters"]
		
		if mpc_verbose:
			print("\nStart Time:", t_s)
			print("Status:", result["status"])
			print("Objective:", result["obj"])
			print("Solve Time:", result["solve_time"])
			print("Iterations:", result["num_iters"])
		
		# Save beam, doses, and penalties for current period.
		status = result["status"]
		beams[t_s] = result["beams"][0]
		doses[t_s] = result["doses"][0]
		
		# Update health for next period.
		h_cur = health_map(F_list[t_s].dot(h_cur) + G_list[t_s].dot(doses[t_s]) + r_list[t_s], t_s)
	
	# Construct full results.
	beams_all = pad_matrix(beams, T_recov)
	doses_all = pad_matrix(doses, T_recov)
	G_list_pad = G_list + T_recov*[np.zeros(G_list[0].shape)]
	health_all = health_prognosis(h_init, T_treat + T_recov, F_list, G_list_pad, r_list, doses_all, health_map)
	obj = dyn_objective(doses, health_all[:(T_treat+1)], patient_rx).value
	return {"obj": obj, "status": status, "num_iters": num_iters, "solve_time": solve_time, "beams": beams_all, "doses": doses_all, "health": health_all}
