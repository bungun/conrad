import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from scipy.special import xlogy
from scipy.optimize import fsolve, root, root_scalar, brute, fmin

def pad_matrix(A, padding, axis = 0):
	m, n = A.shape
	if axis == 0:
		A_pad = np.zeros((m + padding,n))
		A_pad[:m,:] = A
	elif axis == 1:
		A_pad = np.zeros((m, n + padding))
		A_pad[:,:n] = A
	else:
		raise ValueError("axis must be either 0 or 1.")
	return A_pad

def beam_to_dose_block(A_full, indices_or_sections):
	A_blocks = np.split(A_full, indices_or_sections)
	# A_rows = [np.sum(block, axis = 0) for block in A_blocks]
	A_rows = [np.mean(block, axis = 0) for block in A_blocks]
	A = np.row_stack(A_rows)
	return A

# Health prognosis with a given treatment.
def health_prognosis(x_init, growth_mat, T, u_plan = None, damage_mat = None, w_noise = None):
	m = x_init.shape[0]
	x_prog = np.zeros((T+1,m))
	x_prog[0] = x_init
	
	# Defaults to no treatment.
	if u_plan is None and damage_mat is None:
		damage_mat = np.zeros((m,1))
		u_plan = np.zeros((T,1))
	elif not (u_plan is not None and damage_mat is not None):
		raise ValueError("Both u_plan and damage_mat must be provided")
	
	# Defaults to no noise.
	if w_noise is None:
		w_noise = np.zeros((T,m))
	
	for t in range(T):
		x_prog[t+1] = growth_mat.dot(x_prog[t]) - damage_mat.dot(u_plan[t]) + w_noise[t]
	return x_prog

# Health prognosis with a time-invariant treatment.
def health_invariant(x_init, growth_mat, T, u_invar, w_noise = None):
	m = x_init.shape[0]
	if w_noise is None:
		w_noise = -np.tile(u_invar, (T,1))
	else:
		w_noise -= u_invar
	return health_prognosis(x_init, growth_mat, T, w_noise = w_noise)

# Second order system function at (0, y_init), (t_peak, y_peak), (t_decay, \alpha*y_peak), and (\infty, y_init).
def second_order_equal(ty_peak, y_init = 0, t_decay = None, alpha = 0.5, Ns = 100, b_eps = 1e-3, b_scale = 10, verbose = False):
	y_final = y_init
	t_peak, y_peak = ty_peak
	if t_peak <= 0:
		raise ValueError("t_peak must be strictly positive.")
	if t_decay is not None and t_decay <= t_peak:
		raise ValueError("t_decay must be strictly greater than t_peak.")
	if alpha < 0 or alpha > 1:
		raise ValueError("alpha must be in [0,1].")
	
	def a_coeff(b):
		a0 = -b[1]*np.exp(b[0]*t_peak)*(y_peak - y_final)/(b[0] - b[1])
		a1 = -a0
		a2 = y_final
		return [a0, a1, a2]
	
	def Gfun(b):
		if b[0] == b[1]:
			return np.inf
		a = a_coeff(b)
		
		# r0 = a[0] + a[1] + a[2] - y_init
		r1 = a[0]*np.exp(-b[0]*t_peak) + a[1]*np.exp(-b[1]*t_peak) + a[2] - y_peak
		r2 = a[0]*b[0]*np.exp(-b[0]*t_peak) + a[1]*b[1]*np.exp(-b[1]*t_peak)
		if t_decay is not None:
			r3 = a[0]*np.exp(-b[0]*t_decay) + a[1]*np.exp(-b[1]*t_decay) + a[2] - alpha*y_peak
			return np.linalg.norm([r1, r2, r3], 2)
		else:
			return np.linalg.norm([r1, r2], 2)
	
	t_star = 1/t_peak
	rranges = [(b_eps, b_scale*t_star), (b_eps, b_scale*t_star)]
	sol = brute(Gfun, rranges, Ns = Ns, full_output = True, finish = fmin)
	b = sol[0]
	a = a_coeff(b)
	
	h = lambda t: a[0]*np.exp(-b[0]*t) + a[1]*np.exp(-b[1]*t) + a[2]
	hprime = lambda t: -a[0]*b[0]*np.exp(-b[0]*t) - a[1]*b[1]*np.exp(-b[1]*t)
	if verbose:
		print("Initial:", np.abs(h(0) - y_init))
		print("Peak Point:", np.abs(h(t_peak) - y_peak))
		print("Peak Derivative:", np.abs(hprime(t_peak)))
		print("Half-life:", np.abs(h(t_decay) - alpha*y_peak))
	return h

# Second order function at (0, y_init), (t_peak, y_peak), (t_decay, \alpha*y_peak), and (\infty, y_final).
def second_order_unequal(ty_peak, y_init = 0, y_final = 1, t_decay = None, alpha = 0.5, Ns = 100, b_eps = 1e-3, b_scale = 10, verbose = False):
	t_peak, y_peak = ty_peak
	if t_peak <= 0:
		raise ValueError("t_peak must be strictly positive.")
	if t_decay is not None and t_decay <= t_peak:
		raise ValueError("t_decay must be strictly greater than t_peak.")
	if alpha < 0 or alpha > 1:
		raise ValueError("alpha must be in [0,1].")
	if y_init == y_final:
		raise ValueError("y_init cannot be equal to y_final.")
	
	def a_coeff(b):
		a0 = -b[1]*np.exp((b[0] - b[1])*t_peak)*(y_init - y_final)/(b[0] - b[1]*np.exp((b[0] - b[1])*t_peak))
		a2 = y_final
		a1 = y_init - a0 - a2
		return [a0, a1, a2]
	
	def Gbase(b):
		if b[0] == b[1]*np.exp((b[0] - b[1])*t_peak):
			return np.inf
		a = a_coeff(b)
		
		# r0 = a[0] + a[1] + a[2] - y_init
		r1 = a[0]*np.exp(-b[0]*t_peak) + a[1]*np.exp(-b[1]*t_peak) + a[2] - y_peak
		r2 = a[0]*b[0]*np.exp(-b[0]*t_peak) + a[1]*b[1]*np.exp(-b[1]*t_peak)
		if t_decay is not None:
			r3 = a[0]*np.exp(-b[0]*t_decay) + a[1]*np.exp(-b[1]*t_decay) + a[2] - alpha*y_peak
			return np.linalg.norm([r1, r2, r3], 2)
		else:
			return np.linalg.norm([r1, r2], 2)
	
	t_star = 1/t_peak
	f = lambda x: x*np.exp(-x*t_peak)
	if y_init > y_final:
		Gcheck = lambda b: b[0] > b[1] and f(b[0]) > f(b[1]) or b[0] < b[1] and f(b[0]) < f(b[1])
	else:
		Gcheck = lambda b: b[0] > b[1] and f(b[0]) < f(b[1]) or b[0] < b[1] and f(b[0]) > f(b[1])
	Gfun = lambda b: Gbase(b) if Gcheck(b) else np.inf
	
	rranges = [(b_eps, b_scale*t_star), (b_eps, b_scale*t_star)]
	sol = brute(Gfun, rranges, Ns = Ns, full_output = True, finish = fmin)
	b = sol[0]
	a = a_coeff(b)
	
	h = lambda t: a[0]*np.exp(-b[0]*t) + a[1]*np.exp(-b[1]*t) + a[2]
	hprime = lambda t: -a[0]*b[0]*np.exp(-b[0]*t) - a[1]*b[1]*np.exp(-b[1]*t)
	if verbose:
		print("Initial:", np.abs(h(0) - y_init))
		print("Peak Point:", np.abs(h(t_peak) - y_peak))
		print("Peak Derivative:", np.abs(hprime(t_peak)))
		print("Half-life:", np.abs(h(t_decay) - alpha*y_peak))
	return h

# Second order dynamic system curve.
def second_order_curve(ty_peak, y_init = 0, y_final = 0, t_decay = None, alpha = 0.5, T = 100, stepsize = 0.1, tau = 0, Ns = 100, b_eps = 1e-3, b_scale = 10, verbose = False):
	if y_init == y_final:
		h = second_order_equal(ty_peak, y_init, t_decay, alpha = alpha, Ns = Ns, b_eps = b_eps, b_scale = b_scale, verbose = verbose)
	else:
		h = second_order_unequal(ty_peak, y_init, y_final, t_decay, alpha = alpha, Ns = Ns, b_eps = b_eps, b_scale = b_scale, verbose = verbose)
	t_time = np.arange(tau, T + tau, stepsize)
	y_curve = h(t_time - tau)
	return t_time, y_curve
