from conrad.compat import *

import numpy as np

from conrad.defs import cvxpy_var_size
from conrad.optimization.objectives import TreatmentObjective
from conrad.medicine.structure import Structure

class ObjectiveMethods(object):
	@staticmethod
	def normalize(structure):
		if structure.collapsable:
			structure.objective.normalization = 1.
		elif structure.objective.normalization == 1:
			if structure.weighted_size is None:
				raise ValueError(
						'attributes `size` or `voxel_weights` of '
						'argument `structure` must be set to normalize '
						'objective')
			structure.objective.normalization = 1. / structure.weighted_size

	@staticmethod
	def get_weights(structure):
		if structure.collapsable:
			return None
		if structure.size == structure.weighted_size:
			return None
		else:
			return structure.voxel_weights

	@staticmethod
	def eval(structure, y=None, x=None):
		return ObjectiveMethods.primal_eval(structure, y, x)

	@staticmethod
	def primal_eval(structure, y=None, x=None, meow=False):
		if y is not None:
			structure.assign_dose(y)
		elif x is not None:
			structure.calculate_dose(x)

		ObjectiveMethods.normalize(structure)
		y = structure.y if not structure.collapsable else float(
				structure.y_mean)
		weights = None if structure.collapsable else structure.voxel_weights
		return structure.objective.eval(y, weights)

	@staticmethod
	def dual_eval(structure, nu):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		return structure.objective.dual_eval(nu, weights)

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
		size = 1 if structure.collapsable else structure.size
		return structure.objective.primal_expr_pogs(size, weights)

	@staticmethod
	def dual_expr_pogs(structure):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		size = 1 if structure.collapsable else structure.size
		return structure.objective.dual_expr_pogs(size, weights)

	@staticmethod
	def dual_domain_constraints(structure, nu_var):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		return structure.objective.dual_domain_constraints(nu_var, weights)

	@staticmethod
	def dual_domain_constraints_pogs(structure):
		ObjectiveMethods.normalize(structure)
		weights = ObjectiveMethods.get_weights(structure)
		size = 1 if structure.collapsable else structure.size
		return structure.objective.dual_domain_constraints_pogs(size, weights)






