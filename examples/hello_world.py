from conrad import *
from numpy.random import rand

case = Case()
case.anatomy += Structure(0, 'target', True)
case.anatomy += Structure(1, 'avoid', False)

voxels, beams = 1000, 200

# randomly label voxels as 0 (~20%) or 1 (~80%)
voxel_labels = (rand(voxels) > 0.2).astype(int)

dose_matrix = rand(voxels, beams)
FACTOR = 3.
for row, label in enumerate(voxel_labels):
	if label == 0:
		dose_matrix[row, :] *= FACTOR

case.physics.voxel_labels = voxel_labels
case.physics.dose_matrix = dose_matrix

status, run = case.plan()

print('SOLVER FEASIBLE?: {}'.format(status))
print('SOLVE TIME: {}'.format(run.solvetime))
print('NONZERO BEAMS: {}'.format(run.nonzero_beam_count))
print(case.anatomy.dose_summary_string)
