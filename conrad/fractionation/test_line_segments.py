import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from data_utils import *

n = 1000
d = 50
angle = 3*np.pi/4
xlim = np.array([-1, 1])
ylim = np.array([-1, 1])

x_vec = np.linspace(xlim[0], xlim[1], n)
y_vec = np.linspace(ylim[0], ylim[1], n)
X, Y = np.meshgrid(x_vec, y_vec)
# L = line_pixel_length(d, theta, n)
L = line_pixel_length(d, np.pi-angle, n)

segments = line_segments(np.array([angle]), np.array([d]), n, xlim, ylim)
colors = np.array([[0, 1, 0, 1]])
lc = LineCollection(segments, colors = colors, linestyles = "dashed", linewidths = 1)
fig, ax = plt.subplots()
ax.add_collection(lc)

p0 = segments[0,0,:]
p1 = segments[0,1,:]
slope = (p1[1] - p0[1])/(p1[0] - p0[0])

print("Point 0 = ({0},{1})".format(p0[0], p0[1]))
print("Point 1 = ({0},{1})".format(p1[0], p1[1]))
print("Slope = {0}".format(slope))

plt.contourf(X, Y, L, cmap = plt.cm.Blues)
# plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "--r", lw = 1)
plt.hlines(yc, xlim[0], xlim[1], linestyles = "dotted", lw = 1)
plt.vlines(xc, ylim[0], ylim[1], linestyles = "dotted", lw = 1)
plt.show()
