import numpy as np

# Center and rotate counterclockwise by an angle from the x-axis.
def coord_transf(x, y, center = (0,0), angle = 0):
	x0, y0 = center
	xr = (x - x0)*np.cos(angle) - (y - y0)*np.sin(angle)
	yr = (x - x0)*np.sin(angle) + (y - y0)*np.cos(angle)
	return xr, yr
	
# Equation of a circle.
def circle(x, y, center = (0,0), radius = 1):
	return ellipse(x, y, center, (radius, radius))

# Equation of an ellipse.
def ellipse(x, y, center = (0,0), width = (1,1), angle = 0):
	x_width, y_width = width
	xr, yr = coord_transf(x, y, center, angle)
	return (xr/x_width)**2 + (yr/y_width)**2 - 1

# Equation of a cardiod.
def cardioid(x, y, a = 0.5, center = (0,0), angle = 0):
	# xr, yr = coord_transf(x, y, center, angle)
	# return (xr**2 + yr**2)**2 + 4*a*xr*(xr**2 + yr**2) - 4*a**2*yr**2
	return limacon(x, y, -2*a, 2*a, center, angle)

# Equation of a limacon.
def limacon(x, y, a = 1, b = 0, center = (0,0), angle = 0):
	xr, yr = coord_transf(x, y, center, angle)
	return (xr**2 + yr**2 - a*xr)**2 - b**2*(xr**2 + yr**2)

# Generate xy-coordinate pairs from line angles and displacements.
def line_segments(angles, d_vec, xlim = (-1,1), ylim = (-1,1)):
	if np.any(angles < 0) or np.any(angles > np.pi):
		raise ValueError("angles must all be in [0,pi]")
	
	n_angles = len(angles)
	n_offsets = len(d_vec)
	n_lines = n_angles*n_offsets
	
	xc = (xlim[1] + xlim[0])/2
	yc = (ylim[1] + ylim[0])/2
	x_edges = np.zeros((n_lines, 2))
	y_edges = np.zeros((n_lines, 2))
	
	k = 0
	for i in range(n_angles):
		# Slope of line.
		slope = np.tan(np.pi - angles[i])
		
		# Center of line.
		x0 = xc - d_vec*np.sin(np.pi/2 - angles[i])
		y0 = yc + d_vec*np.cos(np.pi/2 - angles[i])
		
		# Endpoints of line.
		for j in range(n_offsets):
			if slope == 0:
				x_edges[k,:] = [xlim[0], y0[j]]
				y_edges[k,:] = [xlim[1], y0[j]]
			elif np.isinf(slope):
				x_edges[k,:] = [x0[j], ylim[0]]
				y_edges[k,:] = [x0[j], ylim[1]]
			else:
				# y - y0 = slope*(x - x0).
				ys = y0[j] + slope*(xlim - x0[j])
				xs = x0[j] + (1/slope)*(ylim - y0[j])
				
				# Save points at edge of grid.
				idx_y = (ys >= ylim[0]) & (ys <= ylim[1])
				idx_x = (xs >= xlim[0]) & (xs <= xlim[1])
				e1 = np.column_stack([xlim, ys])[idx_y]
				e2 = np.column_stack([xs, ylim])[idx_x]
				edges = np.row_stack([e1, e2])
				
				x_edges[k,:] = edges[0]
				y_edges[k,:] = edges[1]
			k = k + 1
	
	coords = np.zeros((2*n_lines,2))
	coords[0::2,:] = x_edges
	coords[1::2,:] = y_edges
	segments = np.split(coords, n_lines)
	return segments

# Construct line integral matrix.
def line_integral_mat(structures, angles = 10, n_bundle = 1, offset = 0.01, *args, **kwargs):
	m_grid, n_grid = structures.shape
	K = np.unique(structures).size
	
	if m_grid != n_grid:
		raise NotImplementedError("Only square grids are supported")
	if np.isscalar(angles):
		angles = np.linspace(0, np.pi, angles+1)[:-1]
	if n_bundle <= 0:
		raise ValueError("n_bundle must be a positive integer")
	if offset < 0:
		raise ValueError("offset must be a nonnegative number")
	
	# A_{kj} = fraction of beam j that falls in region k.
	n_angle = len(angles)
	n_bundle = int(n_bundle)
	n = n_angle*n_bundle
	A = np.zeros((K, n))
	
	# Orthogonal offsets of line from image center (pos = northwest).
	n_half = n_bundle//2
	d_vec = np.arange(-n_half, n_half+1)
	if n_bundle % 2 == 0:
		d_vec = d_vec[:-1]
	d_vec = offset*d_vec
	
	j = 0
	for i in range(n_angle):
		for d in d_vec:
			L = line_pixel_length(d, angles[i], n_grid)
			for k in range(K):
				A[k,j] = np.sum(L[structures == k])
			j = j + 1
	return A, angles, d_vec

def line_pixel_length(d, theta, n):
	"""
	Image reconstruction from line measurements.
	
	Given a grid of n by n square pixels and a line over that grid,
	compute the length of line that goes over each pixel.
	
	Parameters
	----------
	d : displacement of line, i.e., distance of line from center of image, 
		measured in pixel lengths (and orthogonally to line).
	theta : angle of line, measured in radians clockwise from x-axis. 
			Must be between 0 and pi, inclusive.
	n : image size is n by n.
	
	Returns
	-------
	Matrix of size n by n (same as image) with length of the line over 
	each pixel. Most entries will be zero.
	"""
	# For angle in [pi/4,3*pi/4], flip along diagonal (transpose) and call recursively.
	if theta > np.pi/4 and theta < 3*np.pi/4:
		return line_pixel_length(d, np.pi/2-theta, n).T
	
	# For angle in [3*pi/4,pi], redefine line to go in opposite direction.
	if theta > np.pi/2:
		d = -d
		theta = theta - np.pi
	
	# For angle in [-pi/4,0], flip along x-axis (up/down) and call recursively.
	if theta < 0:
		return np.flipud(line_pixel_length(-d, -theta, n))
	
	if theta > np.pi/2 or theta < 0:
		raise ValueError("theta must be in [0,pi]")
	
	L = np.zeros((n,n))
	ct = np.cos(theta)
	st = np.sin(theta)
	
	x0 = n/2 - d*st
	y0 = n/2 + d*ct
	
	y = y0 - x0*st/ct
	jy = int(np.ceil(y))
	dy = (y + n) % 1
	
	for jx in range(n):
		dynext = dy + st/ct
		if dynext < 1:
			if jy >= 1 and jy <= n:
				L[n-jy, jx] = 1/ct
			dy = dynext
		else:
			if jy >= 1 and jy <= n:
				L[n-jy, jx] = (1-dy)/st
			if jy+1 >= 1 and jy + 1 <= n:
				L[n-(jy+1), jx] = (dynext-1)/st
			dy = dynext - 1
			jy = jy + 1
	return L

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

# Check dynamics matrices are correct dimension.
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
		G_list = T*[np.zeros((K,K))]
		doses = np.zeros((T,K))
	elif not (G_list is not None and doses is not None):
		raise ValueError("Both G_list and doses must be provided.")
	if r_list is None:
		r_list = T*[np.zeros(K)]
	
	F_list, G_list, r_list = check_dyn_matrices(F_list, G_list, r_list, K, T, T_recov = 0)
	for t in range(T):
		h_prog[t+1] = health_map(F_list[t].dot(h_prog[t]) + G_list[t].dot(doses[t]) + r_list[t], t)
	return h_prog
