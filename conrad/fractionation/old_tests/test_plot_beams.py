import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from plot_utils import plot_beams

def multiline(xs, ys, c, ax = None, autoscale = False, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # Find axes
    ax = plt.gca() if ax is None else ax

    # Create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # Set coloring of line segments
    lc.set_array(np.asarray(c))

    # Add lines to axes and rescale 
    ax.add_collection(lc)
    if autoscale:
        ax.autoscale()
    return lc

xs = [[np.pi/3, np.pi/3 + np.pi],
      [np.pi/2, np.pi/2 + np.pi]]
ys = [[1, 1],
      [1, 1]]
c = [0, 1]

# fig = plt.figure()
# ax = plt.subplot(111, projection='polar')
# ax = fig.add_subplot(111, projection = "polar")
# lc = multiline(xs, ys, c, cmap='bwr', ax=ax, lw=2)
# axcb = fig.colorbar(lc)
# plt.show()

T = 10
# b = np.array([[25, 50, 100, 150, 200]])
b = np.array([t*np.arange(5, 200, 1) for t in range(T)])
plot_beams(b, stepsize = 4, cmap = "Reds")
