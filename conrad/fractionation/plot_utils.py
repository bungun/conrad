import numpy as np
import matplotlib.pyplot as plt

savepath = "/home/anqi/Dropbox/Research/Fractionation/Figures/"

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
		ax = axs[i] if rows == 1 else axs[int(i / maxcols), i % maxcols]
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
		ax = axs[j] if rows == 1 else axs[int(j / maxcols), j % maxcols]
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

# Plot impulse responses.
def plot_impulse(H, t_range, stepsize = 10, T_treat = None, filename = None):
	m = H.shape[1]
	n = H.shape[2]
	
	fig, axs = plt.subplots(m,n, sharex = True, sharey = True)
	fig.set_size_inches(16,10)
	for i in range(m):
		for j in range(n):
			axs[i,j].plot(t_range, H[:,i,j])
			axs[i,j].set_title("$H_{{{0}{1}}}(t)$".format(i,j))
			
			# Label transition from treatment to recovery period.
			axs[i,j].axvline(x = T_treat, lw = 1, ls = ':', color = "grey")
			xt = np.arange(t_range[0], t_range[-1], stepsize)
			xt = np.append(xt, t_range[-1])
			if T_treat is not None:
				xt = np.append(xt, T_treat)
			axs[i,j].set_xticks(xt)
	plt.suptitle("Impulse Responses $H_{{ij}}(t)$ of Treatment $j$ on Health Status $i$")
	plt.show()
	if filename is not None:
		fig.savefig(savepath + filename, bbox_inches = "tight", dpi = 300)

def plot_impulse_default(H, stepsize = 10, T_treat = None, filename = None):
	plot_impulse(H, range(H.shape[0]), stepsize = stepsize, T_treat = T_treat, filename = filename)
	
# Plot second-order dynamic system curve.
def plot_second_order(t, y, ty_peak, y_init, y_final, t_decay = None, alpha = 0.5, tau = 0, filename = None):
	plt.plot(t, y)
	
	t_peak, y_peak = ty_peak
	plt.axvline(x = t_peak + tau, ls = '--', color = 'orange')
	plt.axhline(y = y_peak, ls = '--', color = 'orange')
	plt.axvline(x = tau, ls = '--', color = 'grey')
	plt.axhline(y = y_init, ls = '--', color = 'grey')
	plt.axhline(y = y_final, ls = '--', color = 'purple')
	
	if t_decay is not None:
		plt.axvline(x = t_decay + tau, ls = '--', color = 'green')
		plt.axhline(y = alpha*y_peak, ls = '--', color = 'green')
		
	plt.show()
	if filename is not None:
		fig.savefig(savepath + filename, bbox_inches = "tight", dpi = 300)
