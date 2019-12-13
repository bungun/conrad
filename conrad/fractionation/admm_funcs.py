import cvxpy
import numpy as np
from cvxpy import *
from multiprocessing import Process, Pipe
from mpc_funcs import dose_penalty, health_penalty, rx_to_constrs

def build_dyn_prob_dose(A_list, patient_rx, T_recov = 0):
	T_treat = len(A_list)
	K, n = A_list[0].shape
	
	# Define variables.
	b = Variable((T_treat,n), pos = True, name = "beams")   # Beams.
	p = Variable((T_treat,K), pos = True, name = "prescribed")   # Prescribed dose.
	d = vstack([A_list[t]*b[t] for t in range(T_treat)])   # Doses.
	
	# Dose penalty function.
	obj = sum([dose_penalty(d_var[t], p_var[t], patient_rx["dose_weights"]) for t in range(T)])
	
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
	constrs = [p_t == patient_rx["dose"][t]]
	
	# Additional dose constraints in period.
	if "dose_constrs" in patient_rx:
		dose_constrs_cur = {"lower": patient_rx["dose_constrs"]["lower"][t], "upper": patient_rx["dose_constrs"]["upper"][t]}
		constrs += rx_to_constrs(d_t, dose_constrs_cur)
	
	prob_t = Problem(Minimize(obj), constrs)
	return prob_t, b_t, p_t, d_t

def build_dyn_prob_health(F, G, h_init, patient_rx, T_treat, T_recov = 0):
	K = h_init.shape[0]
	
	# Define variables.
	h = Variable((T_treat+1,K), name = "health")   # Health statuses.
	d = Variable((T_treat,K), pos = True, name = "doses")
	
	# Health penalty function.
	obj = sum([health_penalty(h_var[t+1], patient_rx["health_weights"]) for t in range(T)])
	
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

def dynamic_treatment_admm(A_list, F, G, h_init, patient_rx, T_recov = 0, health_map = lambda h,t: h, rho = 1.0, max_iter = 1000, *args, **kwargs):
	T_treat = len(A_list)
	
	# Set up dose workers.
	pipes = []
	procs = []
	for t in range(T_treat):
		local, remote = Pipe()
		pipes += [local]
		p_dose_t, b_t, p_t, d_t = build_dyn_prob_dose(A_list[t], patient_rx, T_recov)
		procs += [Process(target=run_dose_worker, args=(remote, p_dose_t, d_t, *args, **kwargs))]
		procs[-1].start()
	
	# Proximal health problem.
	p_health, h, d_tld = build_dyn_prob_health(F, G, h_init, patient_rx, T_treat, T_recov)
	d_new = Parameter(d_tld.shape, value = np.zeros(d_tld.shape))
	u = Parameter(d_tld.shape, value = np.zeros(d_tld.shape))
	penalty = (rho/2)*sum_squares(d_tld - d_new + u)
	prox = p_health + Problem(Minimize(penalty))
	
	# ADMM loop.
	for k in range(max_iter):
		# Collect and stack d_t^k for t = 1,...,T.
		d_rows = [pipe.recv() for pipe in pipes]
		d_new.value = np.row_stack(d_rows)
		
		# Compute and send \tilde d_t^k.
		prox.solve(*args, **kwargs)
		for t in range(T_treat):
			pipe[t].send(d_tld[t].value)
		
		# Receive and update u_t^k for t = 1,...,T.
		u_rows = [pipe.recv() for pipe in pipes]
		u.value = np.row_stack(u_rows)
		
		# TODO: Calculate residuals and check stopping rule.
	
	# TODO: Compute final objective, status, solve_time.
	return {"beams": b.value, "doses": d_tld.value, "health": h.value}

def run_dose_worker(pipe, p, d, *args, **kwargs):
	# Proximal dose problem.
	d_new = Parameter(d.shape, value = np.zeros(d.shape))
	u = Parameter(d.shape, value = np.zeros(d.shape))
	penalty = (rho/2)*sum_squares(d - d_new - u)
	prox = p + Problem(Minimize(penalty))
	
	# ADMM loop.
	while True:
		# Compute and send d_t^k.
		prox.solve(*args. **kwargs)
		pipe.send(d.value)
		
		# Receive \tilde d_t^k.
		d_new.value = pipe.recv()
		
		# Update and send u_t^^k.
		u.value += d_new.value - d.value
		pipe.send(u.value)
