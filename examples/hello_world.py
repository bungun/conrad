"""
Copyright 2016 Baris Ungun, Anqi Fu

This file is part of CONRAD.

CONRAD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CONRAD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CONRAD.  If not, see <http://www.gnu.org/licenses/>.
"""
from conrad import *
from numpy.random import rand

case = Case()

# populate case anatomy
case.anatomy += Structure(0, 'target', True)
case.anatomy += Structure(1, 'avoid', False)

voxels, beams = 1000, 200

# randomly label voxels as 0=target (~20%) or 1=avoid (~80%)
voxel_labels = (rand(voxels) > 0.2).astype(int)

# random dose matrix with target voxels receiving ~3x radiation of non-target
FACTOR = 3.
dose_matrix = rand(voxels, beams)
for row, label in enumerate(voxel_labels):
	if label == 0:
		dose_matrix[row, :] *= FACTOR

# populate case physics
case.physics.voxel_labels = voxel_labels
case.physics.dose_matrix = dose_matrix

status, run = case.plan()

print('SOLVER FEASIBLE?: {}'.format(status))
print('SOLVE TIME: {}'.format(run.solvetime))
print('NONZERO BEAMS: {}'.format(run.nonzero_beam_count))
print(case.anatomy.dose_summary_string)