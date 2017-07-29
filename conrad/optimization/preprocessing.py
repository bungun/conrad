"""
Methods to evaluate/build structure objectives.
"""
"""
Copyright 2016--2017 Baris Ungun, Anqi Fu

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

import numpy as np

from conrad.defs import cvxpy_var_size
from conrad.optimization.objectives import TreatmentObjective
from conrad.medicine.structure import Structure

class ObjectiveMethods(object):
	@staticmethod
	def normalize(structure):
		# scale by global scaling factor / structure cardinality
		if structure.collapsable:
			# |S| * G / |S| = G
			structure.objective.normalization = float(
					structure.objective.global_scaling)
		else:
			# G / |S|
			if structure.weighted_size is None:
				raise ValueError(
						'attributes `size` or `voxel_weights` of '
						'argument `structure` must be set to normalize '
						'objective')
			structure.objective.normalization = float(
					structure.objective.global_scaling /
					structure.weighted_size)

	@staticmethod
	def get_weights(structure):
		if structure.collapsable:
			return None
		if structure.size == structure.weighted_size:
			return None
		else:
			return structure.voxel_weights

	@staticmethod
	def weighted_mean(structure, vector):
		return float(
				np.dot(structure.voxel_weights, vector) /
				np.sum(structure.voxel_weights))

	@staticmethod
	def eval(structure, y=None, x=None):
		return ObjectiveMethods.primal_eval(structure, y, x)

	@staticmethod
	def primal_eval(structure, y=None, x=None):
		if y is not None:
			structure.assign_dose(y)
		elif x is not None:
			structure.calculate_dose(x)

		ObjectiveMethods.normalize(structure)
		y = structure.y if not structure.collapsable else float(
				structure.y_mean)
		if y is None:
			raise ValueError(
					'no dose assigned to structure, cannot evaluate '
					'dose objective')
		weights = None if structure.collapsable else structure.voxel_weights
		return structure.objective.eval(y, weights)

	@staticmethod
	def dual_eval(structure, nu):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		if structure.collapsable and np.size(nu) > 1:
			nu_mean = ObjectiveMethods.weighted_mean(structure, nu)
			return structure.objective.dual_eval(nu_mean, weights)
		return structure.objective.dual_eval(nu, weights)

	@staticmethod
	def in_dual_domain(structure, nu, nu_offset=None, nonnegative=False,
					   reltol=1e-2, abstol=1e-2):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		if nonnegative:
			if np.any(nu < -np.abs(abstol)):
				return False
		return structure.objective.in_dual_domain(
				nu, weights, nu_offset=nu_offset, reltol=reltol, abstol=abstol)

	@staticmethod
	def expr(structure, variable):
		return ObjectiveMethods.primal_expr(structure, variable)

	@staticmethod
	def primal_expr(structure, variable):
		ObjectiveMethods.normalize(structure)
		weights = None if structure.collapsable else structure.voxel_weights
		size = cvxpy_var_size(variable)
		matrix = None
		if structure.collapsable and size != 1:
			matrix = structure.A_mean.reshape((1, structure.A_mean.size))
		elif not structure.collapsable and size != structure.A.shape[0]:
			matrix = structure.A
		if matrix is None:
			return structure.objective.expr(variable, weights)
		else:
			return structure.objective.expr_Ax(matrix, variable, weights)

	@staticmethod
	def dual_expr(structure, nu_var):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		# NB: no adjustment for variable size for collapsable structures
		# since *linear* objective implies dual objective value = 0
		# (affine->constant objective value)
		return structure.objective.dual_expr(nu_var, weights)

	@staticmethod
	def primal_expr_pogs(structure):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		return structure.objective.primal_expr_pogs(
				structure.working_size, weights)

	@staticmethod
	def dual_expr_pogs(structure):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		return structure.objective.dual_expr_pogs(
				structure.working_size, weights)

	@staticmethod
	def dual_domain_constraints(structure, nu_var, nu_offset=None,
								nonnegative=False):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		return structure.objective.dual_domain_constraints(
				nu_var, weights, nu_offset=nu_offset, nonnegative=nonnegative)

	@staticmethod
	def dual_domain_constraints_pogs(structure, nu_offset=None,
									 nonnegative=False):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		return structure.objective.dual_domain_constraints_pogs(
				structure.working_size, weights, nu_offset=nu_offset,
				nonnegative=nonnegative)

	@staticmethod
	def dual_fused_expr_constraints_pogs(structure, nu_offset=None,
										 nonnegative=False):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		return structure.objective.dual_fused_expr_constraints_pogs(
				structure.working_size, weights, nu_offset=nu_offset,
				nonnegative=nonnegative)

	@staticmethod
	def partial_beam_prices(structure, voxel_prices):
		if structure.collapsable:
			if np.size(voxel_prices) > 1:
				voxel_prices = ObjectiveMethods.weighted_mean(
						structure, voxel_prices)
			return structure.A_mean * float(voxel_prices)
		else:
			return structure.A.T.dot(voxel_prices)

	@staticmethod
	def beam_prices(structures, voxel_prices_by_label):
		pbp = lambda s: ObjectiveMethods.partial_beam_prices(
				s, voxel_prices_by_label[s.label])
		return np.add.reduce(listmap(pbp, structures))

	@staticmethod
	def voxel_doses(structures, beam_intensities=None):
		if beam_intensities is not None:
			for s in structures:
				s.calculate_dose(beam_intensities)

		return np.hstack([s.y_mean if s.collapsable else s.y for s in structures])

	@staticmethod
	def apply_joint_scaling(structures):
		weight_scaling = 1.
		dose_scaling = 1.
		for s in structures:
			if s.is_target:
				weight_scaling = max(weight_scaling, float(s.weighted_size))
				dose_scaling = max(dose_scaling, float(s.dose))

		for s in structures:
			s.objective.global_scaling = weight_scaling
		return weight_scaling, dose_scaling