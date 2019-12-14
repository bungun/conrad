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
	p = Variable((T_treat,K), pos = True, name = "prescribed")   # Prescribed dose.
	d = vstack([A_list[t]*b[t] for t in range(T_treat)])   # Doses.
	
	# Dose penalty function.
	obj = sum([dose_penalty(d[t], p[t], patient_rx["dose_weights"]) for t in range(T_treat)])
	
	# Prescribed dose in each period.
	# TODO: Add more bounds on the prescription?
	constrs = [p == patient_rx["dose"]]
	
	# Additional dose constraints.
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(vstack(d), patient_rx["dose_constrs"])
	
	prob = Problem(Minimize(obj), constrs)
	return prob, b, p, d

def build_dyn_prob_dose_period(A, patient_rx, T_recov = 0):
	K, n = A.shape
	
	# Define variables for period.
	b_t = Variable(n, pos = True, name = "beams")   # Beams.
	p_t = Variable(K, pos = True, name = "prescribed")   # Prescribed dose.
	d_t = A*b_t
	
	# Dose penalty current period.
	obj = dose_penalty(d_t, p_t, patient_rx["dose_weights"])
	
	# Prescribed dose in current period.
	# TODO: Add more bounds on the prescription?
	constrs = [p_t == patient_rx["dose"]]
	
	# Additional dose constraints in period.
	if "dose_constrs" in patient_rx:
		constrs += rx_to_constrs(d_t, dose_constrs_cur)
	
	prob_t = Problem(Minimize(obj), constrs)
	return prob_t, b_t, p_t, d_t

def build_dyn_prob_health(F, G, h_init, patient_rx, T_treat, T_recov = 0):
	K = h_init.shape[0]
	
	# Define variables.
	h = Variable((T_treat+1,K), name = "health")   # Health statuses.
	d = Variable((T_treat,K), pos = True, name = "doses")
	
	# Health penalty function.
	obj = sum([health_penalty(h[t+1], patient_rx["health_weights"]) for t in range(T_treat)])
	
	# Health dynamics for treatment stage.
	# constrs = [h[0] == h_init, b >= 0]
	constrs = [h[0] == h_init]
	for t in range(T_treat):
		constrs.append(h[t+1] == F*h[t] + G*d[t])
	
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
	
	prob = Problem(Minimize(obj), constrs)
	return prob, h, d

def dynamic_treatment_admm(A_list, F, G, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, *args, **kwargs):
	# Problem parameters.
	max_iter = kwargs.pop("max_iter", 1000) # Maximum iterations.
	rho = kwargs.pop("rho", 1/10)           # Step size.
	eps_abs = kwargs.pop("eps_abs", 1e-6)   # Absolute stopping tolerance.
	eps_rel = kwargs.pop("eps_rel", 1e-8)   # Relative stopping tolerance.
	
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
		rx_cur["dose"] = patient_rx["dose"][t]
		if "dose_constrs" in patient_rx:
			rx_cur["dose_constrs"] = {"lower": patient_rx["dose_constrs"]["lower"][t], "upper": patient_rx["dose_constrs"]["upper"][t]}
		if "health_constrs" in patient_rx:
			rx_cur["health_constrs"] = {"lower": patient_rx["health_constrs"]["lower"][t], "upper": patient_rx["health_constrs"]["upper"][t]}
			
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target=run_dose_worker, args=(remote, A_list[t], rx_cur, T_recov, rho) + args, kwargs=kwargs)]
		procs[-1].start()
	
	# Proximal health problem.
	prob_health, h, d_tld = build_dyn_prob_health(F, G, h_init, patient_rx, T_treat, T_recov)
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
	while not finished:
		# TODO: Add verbose printout.
		if k % 10 == 0:
			print("Iteration:", k)
            
		# Collect and stack d_t^k for t = 1,...,T.
		d_rows = [pipe.recv() for pipe in pipes]
		d_new.value = np.row_stack(d_rows)
		
		# Compute and send \tilde d_t^k.
		d_tld_prev = np.zeros((T_treat,K)) if k == 0 else d_tld.value
		prox.solve(*args, **kwargs)
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
	
	# Receive final values of b_t^k and p_t^k.
	bp_update = [pipe.recv() for pipe in pipes]
	b_rows, p_rows = map(list, zip(*bp_update))
	b_val = np.row_stack(b_rows)
	p_val = np.row_stack(p_rows)
	
	[proc.terminate() for proc in procs]
	end = time()
	
	# Construct full results.
	b_all = pad_matrix(b_val, T_recov)
	d_all = pad_matrix(d_tld.value, T_recov)
	h_all = health_prognosis(F, h_init, T_treat + T_recov, G, d_all, health_map)
	obj = dyn_objective(d_tld.value, h.value, p_val, patient_rx).value
	
	return {"obj": obj, "status": prox.status, "beams": b_all, "doses": d_all, "health": h_all, "prescribed": p_val, 
			"primal": np.array(r_prim[:k]), "dual": np.array(r_dual[:k]), "num_iters": k, "solve_time": end - start}

def run_dose_worker(pipe, A, patient_rx, T_recov, rho, *args, **kwargs):
	# Construct proximal dose problem.
	prob_dose, b, p, d = build_dyn_prob_dose_period(A, patient_rx, T_recov)
	d_new = Parameter(d.shape, value = np.zeros(d.shape))
	u = Parameter(d.shape, value = np.zeros(d.shape))
	penalty = (rho/2)*sum_squares(d - d_new - u)
	prox = prob_dose + Problem(Minimize(penalty))
	
	# ADMM loop.
	finished = False
	while not finished:
		# Compute and send d_t^k.
		prox.solve(*args, **kwargs)
		pipe.send(d.value)
		
		# Receive \tilde d_t^k.
		d_new.value = pipe.recv()
		
		# Update and send u_t^^k.
		u.value += d_new.value - d.value
		pipe.send(u.value)
		
		# Check if stopped.
		finished = pipe.recv()
	
	# Send final b_t^k and p_t^k.
	pipe.send((b.value, p.value))
