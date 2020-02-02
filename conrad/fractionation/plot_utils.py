import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

savepath = "/home/anqi/Dropbox/Research/Fractionation/Figures/"

# Plot beams.
def plot_beams(b, theta = None, stepsize = 10, maxcols = 5, standardize = False, filename = None, *args, **kwargs):
	T = b.shape[0]
	m = b.shape[1]
	
	if theta is None:
		theta = np.linspace(0, np.pi, m+1)[:-1]
		# theta = np.linspace(0, 2*np.pi, m+1)[:-1]
	if theta.shape[0] != m:
		raise ValueError("theta must be an array of length {0}".format(m))
	if standardize:
		b_mean = np.tile(np.mean(b, axis = 1), (m,1)).T
		b_std = np.tile(np.std(b, axis = 1), (m,1)).T
		b = (b - b_mean)/b_std
	
	T_steps = int(np.floor(T / stepsize)) + 1
	rows = 1 if T_steps <= maxcols else int(np.ceil(T_steps / maxcols))
	cols = min(T_steps, maxcols)
	
	fig, axs = plt.subplots(rows, cols, subplot_kw = dict(projection = "polar"))
	fig.set_size_inches(16,8)
	
	# Create collection of beams.
	arr = np.ones((2*m, 2))
	arr[0::2,0] = theta
	# arr[1::2] = 0
	arr[1::2,0] = theta + np.pi
	segments = np.split(arr, m)
	lc = LineCollection(segments, *args, **kwargs)
	
	for t in np.arange(0, T, stepsize):
		if rows == 1:
			ax = axs if cols == 1 else axs[t]
		else:
			ax = axs[int(t / maxcols), t % maxcols]
		
		# Set colors based on beam intensity.
		lc.set_array(np.asarray(b[t]))
		ax.add_collection(lc)
		ax.set_yticklabels([])
		ax.set_theta_zero_location('N')
		ax.set_title("$b({0})$".format(t))
	
	fig.colorbar(lc)
	plt.suptitle("Beam Intensities vs. Time")
	plt.show()
	if filename is not None:
		fig.savefig(savepath + filename, bbox_inches = "tight", dpi = 300)

# Plot health curves.
def plot_health(x, curves = {}, stepsize = 10, maxcols = 5, T_treat = None, filename = None):
	T = x.shape[0] - 1
	m = x.shape[1] 
	
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
		ltreat, = ax.plot(range(T+1), x[:,i], label = "Treated")
		handles = [ltreat]
		for label, curve in curves.items():
			lcurve, = ax.plot(range(T+1), curve[:,i], ls = '--', label = label)
			handles += [lcurve]
		# lnone, = ax.plot(range(T+1), x_prog[:,i], ls = '--', color = "red")
		# ax.set_title("$x_{{{0}}}(t)$".format(i))
		ax.set_title("$h_{{{0}}}(t)$".format(i))
		
		# Label transition from treatment to recovery period.
		ax.axvline(x = T_treat, lw = 1, ls = ':', color = "grey")
		xt = np.arange(0, T, stepsize)
		xt = np.append(xt, T)
		if T_treat is not None:
			xt = np.append(xt, T_treat)
		ax.set_xticks(xt)
	
	for col in range(left):
		axs[rows-1, maxcols-1-col].set_axis_off()
	
	fig.legend(handles = handles, loc = "center right", borderaxespad = 1)
	# fig.legend(handles = [ltreat, lnone], labels = ["Treated", "Untreated"], loc = "center right", borderaxespad = 1)
	plt.suptitle("Health Status vs. Time")
	plt.show()
	if filename is not None:
		fig.savefig(savepath + filename, bbox_inches = "tight", dpi = 300)

# Plot treatment curves.
def plot_treatment(u, stepsize = 10, maxcols = 5, T_treat = None, filename = None):
	T = u.shape[0] - 1
	n = u.shape[1]
	
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
		ax.plot(range(T+1), u[:,j])
		# ax.set_title("$u_{{{0}}}(t)$".format(j))
		ax.set_title("$d_{{{0}}}(t)$".format(j))
		
		# Label transition from treatment to recovery period.
		ax.axvline(x = T_treat, lw = 1, ls = ':', color = "grey")
		xt = np.arange(0, T, stepsize)
		xt = np.append(xt, T)
		if T_treat is not None:
			xt = np.append(xt, T_treat)
		ax.set_xticks(xt)
	
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
