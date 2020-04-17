import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from data_utils import *

n = 1000
d = 50
angle = 3*np.pi/4
xlim = np.array([-1, 1])
ylim = np.array([-1, 1])
xc = (xlim[0] + xlim[1])/2
yc = (ylim[0] + ylim[1])/2

x_vec = np.linspace(xlim[0], xlim[1], n)
y_vec = np.linspace(ylim[0], ylim[1], n)
X, Y = np.meshgrid(x_vec, y_vec)
# L = line_pixel_length(d, theta, n)
L = line_pixel_length(d, np.pi-angle, n)

x_scale = (xlim[1] - xlim[0])/n   # (x_max - x_min)/(x_len_pixels) coord/pixel
y_scale = (ylim[1] - ylim[0])/n   # (y_max - y_min)/(y_len_pixels) coord/pixel
dydx_scale = (ylim[1] - ylim[0])/(xlim[1] - xlim[0])

slope = dydx_scale*np.tan(angle)
x0 = xc - d*x_scale*np.sin(angle)
y0 = yc + d*y_scale*np.cos(angle)
print("Slope = {0}".format(slope))
print("Center = ({0}, {1})".format(x0, y0))

if slope == 0:
	p0 = [xlim[0], y0]
	p1 = [xlim[1], y0]
elif np.isinf(slope):
	p0 = [x0, ylim[0]]
	p1 = [x0, ylim[1]]
else:
	# y - y0 = slope*(x - x0).
	ys = y0 + slope*(xlim - x0)
	xs = x0 + (1/slope)*(ylim - y0)
	
	# Save points at edge of grid.
	idx_y = (ys >= ylim[0]) & (ys <= ylim[1])
	idx_x = (xs >= xlim[0]) & (xs <= xlim[1])
	e1 = np.column_stack([xlim, ys])[idx_y]
	e2 = np.column_stack([xs, ylim])[idx_x]
	edges = np.row_stack([e1, e2])
	
	p0 = edges[0]
	p1 = edges[1]

segments = line_segments(np.array([angle]), np.array([d]), n, xlim, ylim)
colors = np.array([[0, 1, 0, 1]])
lc = LineCollection(segments, colors = colors, linestyles = "dashed", linewidths = 1)
fig, ax = plt.subplots()
ax.add_collection(lc)
# print(segments)

plt.contourf(X, Y, L, cmap = plt.cm.Blues)
# plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "--r", lw = 1)
plt.hlines(yc, xlim[0], xlim[1], linestyles = "dotted", lw = 1)
plt.vlines(xc, ylim[0], ylim[1], linestyles = "dotted", lw = 1)
plt.show()
