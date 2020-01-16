import numpy as np

# Pad matrix with zeros.
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

# Block average rows of dose influence matrix.
def beam_to_dose_block(A_full, indices_or_sections):
	A_blocks = np.split(A_full, indices_or_sections)
	# A_rows = [np.sum(block, axis = 0) for block in A_blocks]
	A_rows = [np.mean(block, axis = 0) for block in A_blocks]
	A = np.row_stack(A_rows)
	return A

def check_dyn_matrices(F_list, G_list, r_list, K, T_treat, T_recov = 0):
	T_total = T_treat + T_recov
	if not isinstance(F_list, list):
		F_list = T_total*[F_list]
	if not isinstance(G_list, list):
		G_list = T_treat*[G_list]
	if not isinstance(r_list, list):
		r_list = T_total*[r_list]
	
	if len(F_list) != T_total:
		raise ValueError("F_list must be a list of length {0}".format(T_total))
	if len(G_list) != T_treat:
		raise ValueError("G_list must be a list of length {0}".format(T_treat))
	if len(r_list) != T_total:
		raise ValueError("r_list must be a list of length {0}".format(T_total))
	
	for F in F_list:
		if F.shape != (K,K):
			raise ValueError("F_t must have dimensions ({0},{0})".format(K))
	for G in G_list:
		if G.shape != (K,K):
			raise ValueError("G_t must have dimensions ({0},{0})".format(K))
	for r in r_list:
		if r.shape != (K,) and r.shape != (K,1):
			raise ValueError("r_t must have dimensions ({K},)".format(K))
	return F_list, G_list, r_list

# Health prognosis with a given treatment.
def health_prognosis(h_init, T, F_list, G_list = None, r_list = None, doses = None, health_map = lambda h,t: h):
	K = h_init.shape[0]
	h_prog = np.zeros((T+1,K))
	h_prog[0] = h_init
	
	# Defaults to no treatment.
	if G_list is None and doses is None:
		G_list = T*[np.zeros((K,1))]
		doses = np.zeros((T,1))
	elif not (G_list is not None and doses is not None):
		raise ValueError("Both G_list and doses must be provided.")
	if r_list is None:
		r_list = T*[np.zeros(K)]
	
	if not isinstance(F_list, list):
		F_list = T*[F_list]
	if not isinstance(G_list, list):
		G_list = T*[G_list]
	if not isinstance(r_list, list):
		r_list = T*[r_list]
	
	if len(F_list) != T:
		raise ValueError("F_list must be a list of length {0}".format(T))
	if len(G_list) != T:
		raise ValueError("G_list must be a list of length {0}".format(T))
	if len(r_list) != T:
		raise ValueError("r_list must be a list of length {0}".format(T))
	
	for t in range(T):
		h_prog[t+1] = health_map(F_list[t].dot(h_prog[t]) + G_list[t].dot(doses[t]) + r_list[t], t)
	return h_prog
