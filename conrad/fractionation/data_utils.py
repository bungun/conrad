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

# Health prognosis with a given treatment.
def health_prognosis(F, h_init, T, G = None, r = None, doses = None, health_map = lambda h,t: h):
	K = h_init.shape[0]
	h_prog = np.zeros((T+1,K))
	h_prog[0] = h_init
	
	# Defaults to no treatment.
	if G is None and doses is None:
		G = np.zeros((K,1))
		doses = np.zeros((T,1))
	elif not (doses is not None and G is not None):
		raise ValueError("Both G and doses must be provided.")
	if r is None:
		r = np.zeros(K)
	
	for t in range(T):
		h_prog[t+1] = health_map(F.dot(h_prog[t]) + G.dot(doses[t]) + r, t)
	return h_prog
