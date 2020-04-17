import numpy as np
import matplotlib.pyplot as plt
from data_utils import *
from example_utils import *
from plot_utils import plot_structures

# Display structures on a polar grid.
n = 1000
x_grid, y_grid, regions = simple_structures(n, n)
struct_kw = simple_colormap(one_idx = True)
plot_structures(x_grid, y_grid, regions, title = "Anatomical Structures", one_idx = True, **struct_kw)

A, angles, d_vec = line_integral_mat(regions, angles = 4, n_bundle = 1, offset = 0)
print("Angles:", angles)
print("Offsets:", d_vec)

j = 3
print("Pixel Length for Beam Angle = {0}".format(angles[j]))
for k in range(A.shape[0]):
	print("Structure {0} = {1}".format(k+1, A[k,j]))
