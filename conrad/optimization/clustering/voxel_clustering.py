"""
TODO: DOCSTRING
"""
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
from conrad.compat import *

import abc

from conrad.optimization.clustering.clustering_base import ClusteredProblem

@add_metaclass(abc.ABCMeta)
class VoxelClusteredProblem(ClusteredProblem):
	def __init__(self, case, reference_frame_name, clustered_frame_name):
		ClusteredProblem.__init__(
				self, case, reference_frame_name, clustered_frame_name)

		self.__cluster_mapping = case.physics.retrieve_frame_mapping(
				self.reference_frame, self.clustered_frame).voxel_map

	@property
	def cluster_mapping(self):
		return self.__cluster_mapping

class UnconstrainedVoxelClusteredProblem(VoxelClusteredProblem):
	def __init__(self, case, reference_frame_name, clustered_frame_name):
		VoxelClusteredProblem.__init__(
				self, case, reference_frame_name, clustered_frame_name)

	def solve_and_bound_clustered_problem(self, **solver_options):
		self.methods.apply_joint_scaling(self.reference_anatomy)
		self.methods.apply_joint_scaling(self.clustered_anatomy)

		case = self.case

		_, run = self.case.plan(frame=self.clustered_frame, **solver_options)

		x_star_vclu = run.output.x

		obj_lb = self.dose_objective_primal_eval(self.clustered_anatomy)
		self.reference_anatomy.calculate_doses(x_star_vclu)
		obj_ub = self.dose_objective_primal_eval(self.reference_anatomy)

		run.output.optimal_variables['x_full'] = x_star_vclu
		run.output.solver_info['primal_solve_time'] = float(
				case.problem.solver.solvetime + case.problem.solver.setuptime)

		return obj_ub, obj_lb, run



# class ConstrainedVoxelClusteredProblem(VoxelClusteredProblem):
	# pass

# class RowClusteringMethods(object):
# 	@staticmethod
# 	def cluster(structure, desired_compression):
# 		pass

# 	@staticmethod
# 	def collapse(structure):
# 		pass

# 	@staticmethod
# 	def compress_anatomy(case, desired_compression):
# 		pass
# 		# target compression in (int, float, dict)

# 	@staticmethod
# 	def solve_compressed(case, full_frame, compressed_frame):
# 		pass
# 		# return {x: --, y: --, upper_bound: --, lower_bound: --}
