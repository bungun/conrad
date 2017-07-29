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
import numpy as np

from conrad.case import Case
from conrad.optimization.preprocessing import ObjectiveMethods

@add_metaclass(abc.ABCMeta)
class ClusteredProblem(object):
	def __init__(self, case, reference_frame_name, clustered_frame_name):
		if not isinstance(case, Case):
			raise TypeError(
					'argument `case` must be of type {}'.format(Case))
		self.__case = case
		for frame_name in (reference_frame_name, clustered_frame_name):
			if not frame_name in case.physics.available_frames:
				raise ValueError(
						'case has no attached frame named {}'
						''.format(frame_name))
		self.__reference_frame = reference_frame_name
		self.__clustered_frame = clustered_frame_name
		self.__reference_anatomy = None
		self.__clustered_anatomy = None
		self.methods = ObjectiveMethods

		case.load_physics_to_anatomy(frame=self.reference_frame)
		case.load_physics_to_anatomy(frame=self.clustered_frame)

	@property
	def case(self):
		return self.__case

	@property
	def reference_frame(self):
		return self.__reference_frame

	@property
	def clustered_frame(self):
		return self.__clustered_frame

	@property
	def reference_anatomy(self):
		return self.case.anatomy_for_frame(self.reference_frame)

	@property
	def clustered_anatomy(self):
		return self.case.anatomy_for_frame(self.clustered_frame)

	@property
	def reference_problem(self):
		return self.case.problem_for_frame(self.reference_frame)

	@property
	def clustered_problem(self):
		return self.case.problem_for_frame(self.clustered_frame)

	def dose_objective_primal_eval(self, anatomy, voxel_doses=None,
								   beam_intensities=None):
		doses = {s.label: None for s in anatomy}
		if voxel_doses:
			doses.update(voxel_doses)
		p_eval = lambda s: self.methods.primal_eval(
				s, doses[s.label], beam_intensities)
		return sum(listmap(p_eval, anatomy))

	def split_voxel_prices(self, voxel_prices, structure_order,
						   rows_per_structure):
		voxel_price_dict = {}
		offset = 0
		for label in structure_order:
			size = rows_per_structure[label]
			voxel_price_dict[label] = voxel_prices[offset : offset + size]
			offset += size
		return voxel_price_dict

	def join_voxel_prices(self, voxel_price_dict, structure_order=None):
		if structure_order is None:
			structure_order = self.reference_anatomy.label_order
		return np.hstack([voxel_price_dict[label] for label in structure_order])

	def dose_objective_dual_eval(self, anatomy, voxel_price_dict):
		return sum(listmap(
				lambda s: self.methods.dual_eval(s, voxel_price_dict[s.label]),
				anatomy))

	def dose_objective_dual_eval_vec(self, anatomy, voxel_prices):
		voxel_price_dict = self.split_voxel_prices(
				voxel_prices, anatomy.label_order, anatomy.working_sizes)
		return self.dose_objective_dual_eval(anatomy, voxel_price_dict)

	def update_voxel_prices(self, voxel_price_dict, voxel_price_update):
		for label in voxel_price_dict:
			voxel_price_dict[label] += voxel_price_update[label]


	def clear(self):
		self.reference_problem.clear()
		self.clustered_problem.clear()

	@abc.abstractmethod
	def solve_and_bound_clustered_problem(self, **solver_options):
		raise NotImplementedError

	def plan(self, **solver_options):
		ub, lb, run = self.solve_and_bound_clustered_problem(**solver_options)
		run.output.solver_info['upper_bound'] = ub
		run.output.solver_info['lower_bound'] = lb
		run.output.solver_info['suboptimality'] = 100. * (ub - lb) / lb
		return run

