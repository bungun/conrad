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

from conrad.optimization.clustering_base import ClusteredProblem

@add_metaclass(abc.ABCMeta)
class VoxelClusteredProblem(ClusteredProblem):
	def __init__(self, case, reference_frame_name, clustered_frame_name):
		ClusteredProblem.__init__(
				self, case, reference_frame_name, clustered_frame_name)

class UnconstrainedVoxelClusteredProblem(VoxelClusteredProblem):
	def __init__(self, case, reference_frame_name, clustered_frame_name):
		VoxelClusteredProblem.__init__(
				self, case, reference_frame_name, clustered_frame_name)

	def clear(self):
		self.__case.problem.solver.clear()

	def solve_and_bound_clustered_problem(self, case, reference_anatomy,
										  **solver_options):
		_, run = self.case.plan(**solver_options)

		x_star_vclu = run.output.x

		reference_anatomy.calculate_doses(x_star_vclu)
		obj_ub = self.dose_objective_primal_eval(reference_anatomy)
		obj_lb = self.dose_objective_primal_eval(case.anatomy)

		run.output.optimal_variables['x_full'] = x_star_vclu
		run.output.solver_info['primal_solve_time'] = float(
				case.problem.solver.solvetime + case.problem.solver.setuptime)

		return obj_ub, obj_lb, run

	def cluster_and_bound(self,  **solver_options):
		self.reload_clustered_frame()
		return self.solve_and_bound_clustered_problem(
				self.case, self.reference_anatomy, **solver_options)

# class ConstrainedVoxelClusteredProblem(VoxelClusteredProblem):
	# pass