import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

savepath = "/home/anqi/Dropbox/Research/Fractionation/Figures/"

# Extract subset of a colormap.
# http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
def truncate_cmap(cmap, minval = 0.0, maxval = 1.0, n = 100):
    cmap_trunc = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n = cmap.name, a = minval, b = maxval),
        cmap(np.linspace(minval, maxval, n)))
    return cmap_trunc

# Modify colormap to enable transparency in smaller values.
def transp_cmap(cmap):
	cmap_transp = cmap(np.arange(cmap.N))
	cmap_transp[:,-1] = np.linspace(0, 1, cmap.N)
	cmap_transp = ListedColormap(cmap_transp)
	return cmap_transp

# Plot structures.
def plot_structures(theta, r, structures, filename = None, *args, **kwargs):
	m, n = theta.shape
	if r.shape != (m,n):
		raise ValueError("r must have dimensions ({0},{1})".format(m,n))
	if structures.shape != (m,n):
		raise ValueError("structures must have dimensions ({0},{1})".format(m,n))
	
	plt.figure(figsize = (10,8))
	ax = plt.subplot(111, projection = "polar")
	
	labels = np.unique(structures)
	lmin = np.min(labels)
	lmax = np.max(labels)
	levels = np.arange(lmin, lmax + 2) - 0.5
	
	ctf = ax.contourf(theta, r, structures, levels = levels, *args, **kwargs)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_theta_zero_location("N")
	
	plt.title("Anatomical Structures")
	plt.colorbar(ctf, ticks = np.arange(lmin, lmax + 1))
	plt.show()
	if filename is not None:
		fig.savefig(savepath + filename, bbox_inches = "tight", dpi = 300)

# Plot beams.
def plot_beams(b, theta = None, stepsize = 10, maxcols = 5, standardize = False, filename = None, structures = None, struct_kw = dict(), *args, **kwargs):
	T = b.shape[0]
	n = b.shape[1]
	
	if theta is None:
		theta = np.linspace(0, np.pi, n+1)[:-1]
		# theta = np.linspace(0, 2*np.pi, n+1)[:-1]
	if theta.shape[0] != n:
		raise ValueError("theta must be an array of length {0}".format(n))
	if standardize:
		b_mean = np.tile(np.mean(b, axis = 1), (n,1)).T
		b_std = np.tile(np.std(b, axis = 1), (n,1)).T
		b = (b - b_mean)/b_std
	if structures is not None:
		if len(structures) != 3:
			raise ValueError("structures should be a tuple of (theta, r, labels)")
		struct_theta, struct_r, struct_labels = structures
		
		labels = np.unique(struct_labels)
		lmin = np.min(labels)
		lmax = np.max(labels)
		struct_levels = np.arange(lmin, lmax + 2) - 0.5
	
	T_grid = np.arange(0, T, stepsize)
	if T_grid[-1] != T-1:
		T_grid = np.append(T_grid, T-1)   # Always include last time period.
	T_steps = len(T_grid)
	rows = 1 if T_steps <= maxcols else int(np.ceil(T_steps / maxcols))
	cols = min(T_steps, maxcols)
	
	fig, axs = plt.subplots(rows, cols, subplot_kw = dict(projection = "polar"))
	fig.set_size_inches(16,8)
	
	# Create collection of beams.
	arr = np.ones((2*n, 2))
	arr[0::2,0] = theta
	# arr[1::2] = 0
	arr[1::2,0] = theta + np.pi
	segments = np.split(arr, n)
	
	t = 0
	for t_step in range(T_steps):
		if rows == 1:
			ax = axs if cols == 1 else axs[t_step]
		else:
			ax = axs[int(t_step / maxcols), t_step % maxcols]
		
		# Plot anatomical structures.
		if structures is not None:
			ctf = ax.contourf(struct_theta, struct_r, struct_labels, levels = struct_levels, **struct_kw)
		
		# Set colors based on beam intensity.
		lc = LineCollection(segments, *args, **kwargs)
		lc.set_array(np.asarray(b[t]))
		ax.add_collection(lc)
		
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_theta_zero_location('N')
		ax.set_title("$b({0})$".format(t+1))
		t = min(t + stepsize, T-1)
	
	# Display colorbar for structures by label.
	if structures is not None:
		fig.subplots_adjust(left = 0.2)
		cax_left = fig.add_axes([0.125, 0.15, 0.02, 0.7])
		fig.colorbar(ctf, cax = cax_left, ticks = np.arange(lmin, lmax + 1), label = "Structure Label")
		cax_left.yaxis.set_label_position("left")
	
	# Display colorbar for entire range of intensities.
	fig.subplots_adjust(right = 0.8)
	cax_right = fig.add_axes([0.85, 0.15, 0.02, 0.7])
	lc = LineCollection(2*[np.zeros((2,2))], *args, **kwargs)
	lc.set_array(np.array([np.min(b), np.max(b)]))
	fig.colorbar(lc, cax = cax_right, label = "Beam Intensity")
	cax_right.yaxis.set_label_position("left")
	
	plt.suptitle("Beam Intensities vs. Time")
	plt.show()
	if filename is not None:
		fig.savefig(savepath + filename, bbox_inches = "tight", dpi = 300)

# Plot health curves.
def plot_health(h, curves = {}, stepsize = 10, maxcols = 5, T_treat = None, bounds = None, filename = None):
	T = h.shape[0] - 1
	m = h.shape[1]
	
	if bounds is not None:
		lower, upper = bounds
	else:
		lower = upper = None
	
	rows = 1 if m <= maxcols else int(np.ceil(m / maxcols))
	cols = min(m, maxcols)
	left = rows*cols - m
	
	fig, axs = plt.subplots(rows, cols, sharey = True)
	fig.set_size_inches(16,8)
	for i in range(m):
		if rows == 1:
			ax = axs if cols == 1 else axs[i]
		else:
			ax = axs[int(i / maxcols), i % maxcols]
		ltreat, = ax.plot(range(T+1), h[:,i], label = "Treated")
		handles = [ltreat]
		for label, curve in curves.items():
			lcurve, = ax.plot(range(T+1), curve[:,i], label = label)
			handles += [lcurve]
		# lnone, = ax.plot(range(T+1), x_prog[:,i], ls = '--', color = "red")
		# ax.set_title("$x_{{{0}}}(t)$".format(i))
		ax.set_title("$h_{{{0}}}(t)$".format(i))
		
		# Label transition from treatment to recovery period.
		xt = np.arange(0, T, stepsize)
		xt = np.append(xt, T)
		if T_treat is not None:
			ax.axvline(x = T_treat, lw = 1, ls = ':', color = "grey")
			xt = np.append(xt, T_treat)
		ax.set_xticks(xt)
		
		# Plot lower and upper bounds on h(t) for t = 1,...,T.
		if lower is not None:
			ax.plot(range(1,T+1), lower[:,i], lw = 1, ls = "--", color = "cornflowerblue")
		if upper is not None:
			ax.plot(range(1,T+1), upper[:,i], lw = 1, ls = "--", color = "cornflowerblue")
	
	for col in range(left):
		axs[rows-1, maxcols-1-col].set_axis_off()
	
	fig.legend(handles = handles, loc = "center right", borderaxespad = 1)
	# fig.legend(handles = [ltreat, lnone], labels = ["Treated", "Untreated"], loc = "center right", borderaxespad = 1)
	plt.suptitle("Health Status vs. Time")
	plt.show()
	if filename is not None:
		fig.savefig(savepath + filename, bbox_inches = "tight", dpi = 300)

# Plot treatment curves.
def plot_treatment(d, stepsize = 10, maxcols = 5, T_treat = None, bounds = None, filename = None):
	T = d.shape[0]
	n = d.shape[1]
	
	if bounds is not None:
		lower, upper = bounds
	else:
		lower = upper = None
	
	rows = 1 if n <= maxcols else int(np.ceil(n / maxcols))
	cols = min(n, maxcols)
	left = rows*cols - n
	
	fig, axs = plt.subplots(rows, cols, sharey = True)
	fig.set_size_inches(16,8)
	for j in range(n):
		if rows == 1:
			ax = axs if cols == 1 else axs[j]
		else:
			ax = axs[int(j / maxcols), j % maxcols]
		ax.plot(range(1,T+1), d[:,j])
		# ax.set_title("$u_{{{0}}}(t)$".format(j))
		ax.set_title("$d_{{{0}}}(t)$".format(j))
		
		# Label transition from treatment to recovery period.
		xt = np.arange(1, T, stepsize)
		xt = np.append(xt, T)
		if T_treat is not None:
			ax.axvline(x = T_treat, lw = 1, ls = ':', color = "grey")
			xt = np.append(xt, T_treat)
		ax.set_xticks(xt)
		
		# Plot lower and upper bounds on d(t) for t = 1,...,T.
		if lower is not None:
			ax.plot(range(1,T+1), lower[:,j], lw = 1, ls = "--", color = "cornflowerblue")
		if upper is not None:
			ax.plot(range(1,T+1), upper[:,j], lw = 1, ls = "--", color = "cornflowerblue")
	
	for col in range(left):
		axs[rows-1, maxcols-1-col].set_axis_off()
	
	plt.suptitle("Treatment Dose vs. Time")
	plt.show()
	if filename is not None:
		fig.savefig(savepath + filename, bbox_inches = "tight", dpi = 300)

# Plot primal and dual residuals.
def plot_residuals(r_primal, r_dual, normalize = False, show = True, title = None, semilogy = False, filename = None, *args, **kwargs):
	if normalize:
		r_primal = r_primal/r_primal[0] if r_primal[0] != 0 else r_primal
		r_dual = r_dual/r_dual[0] if r_dual[0] != 0 else r_dual
	
	fig = plt.figure()
	fig.set_size_inches(12,8)
	if semilogy:
		plt.semilogy(range(len(r_primal)), r_primal, label = "Primal", *args, **kwargs)
		plt.semilogy(range(len(r_dual)), r_dual, label = "Dual", *args, **kwargs)
	else:
		plt.plot(range(len(r_primal)), r_primal, label = "Primal", *args, **kwargs)
		plt.plot(range(len(r_dual)), r_dual, label = "Dual", *args, **kwargs)

	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("Residual")
	
	if title:
		plt.title(title)
	if show:
		plt.show()
	if filename is not None:
		fig.savefig(savepath + filename, bbox_inches = "tight", dpi = 300)
