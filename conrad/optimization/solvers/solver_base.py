"""
Define base class for interfaces between :mod:`conrad` and Python
modules with convex solver interfaces, e.g., :mod:`cvxpy`.

Attributes:
	GAMMA_DEFAULT (:obj:`float`): Default scaling to apply to objective
		term penalizing weighted sum of slack variables, when dose
		constraint slacks are allowed.
	RELTOL_DEFAULT (:obj:`float`): Default relative tolerance for
		solver.
	ABSTOL_DEFAULT (:obj:`float`): Default absolute tolerance for
		solver.
	VERBOSE_DEFAULT (:obj:`int`): Default solver verbosity.
	MAXITER_DEFAULT (:obj:`int`): Default maximum solver iterations.
	INDIRECT_DEFAULT (:obj:`bool`): Default solver mode (applies to SCS
		only).
	GPU_DEFAULT (:obj:`bool`): Default solver device (applies to SCS,
		POGS).
	PRIORITY_1 (:obj:`int`): Penalty scaling for slack on high priority
		dose constraints.
	PRIORITY_2 (:obj:`int`): Penalty scaling for slack on medium
		priority dose constraints.
	PRIORITY_3 (:obj:`int`): Penalty scaling for slack on low priority
		dose constraints.
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

import numpy as np
import abc

from conrad.optimization.preprocessing import ObjectiveMethods

GAMMA_DEFAULT = 1e-2
RELTOL_DEFAULT = 1e-3
ABSTOL_DEFAULT = 1e-4
VERBOSE_DEFAULT = 1
MAXITER_DEFAULT = 2000
INDIRECT_DEFAULT = False
GPU_DEFAULT = False

PRIORITY_1 = 9
PRIORITY_2 = 4
PRIORITY_3 = 1

PROJECTOR_POGS_DENSE_DIRECT = 'cholesky(identity + gramian)'
PROJECTOR_INDIRECT = 'indirect'

class Solver(object):
	"""
	Base class for translating :mod:`conrad` planning requests to convex
	problems.

	Attributes:
		use_2pass (:obj:`bool`): When ``True``, enables exact
			percentile-type dose constraints to be built (other
			conditions must hold).
		use_slack (:obj:`bool`): When ``True``, dose constraints are
			built with slack.
		dvh_vars (:obj:`dict`): Dictionary, keyed by constraint ID, of
			values for slope variables associated with the convex
			restriction of each percentile-type dose constraint in the
			problem.
		slack_vars (:obj:`dict`): Dictionary, keyed by constraint ID, of
			values for slack variables associated with each dose
			constraint in the problem.
		feasible (:obj:`bool`): ``True`` if most recent optimization run
			was feasible.
	"""
	def __init__(self):
		"""
		Initialize solver with default settings.

		Arguments:
			None
		"""
		self.use_2pass = False
		self.use_slack = True
		self.__x = None
		self.__gamma = GAMMA_DEFAULT
		self.dvh_vars = {}
		self.slack_vars = {}
		self.feasible = False
		self.__global_weight_scaling = 1.
		self.__global_dose_scaling = 1.

	@property
	def gamma(self):
		""" Scaling to apply to weighted sum of slack variables. """
		return self.__gamma

	@gamma.setter
	def gamma(self, gamma):
		if gamma:
			self.__gamma = float(gamma)

	@property
	def global_weight_scaling(self):
		return self.__global_weight_scaling

	@property
	def global_dose_scaling(self):
		return self.__global_dose_scaling

	def gamma_prioritized(self, priority):
		"""
		Calculate penalty scaling for slack variable.

		Arguments:
			priority (:obj:int): Constraint priority, should be ``1``,
				``2`` or ``3``.

		Returns:
			float: Constraint-specific penalty obtained by multiplying
			master scaling for slack variable penalties
			(:attr:`Solver.gamma`) times priority-based scaling.

		Raises:
			ValueError: If ``priority`` is not one of ``1``, ``2`` or
				``3``.

		"""
		priority = int(priority)
		if priority == 1:
			return self.gamma * PRIORITY_1
		elif priority == 2:
			return self.gamma * PRIORITY_2
		elif priority == 3:
			return self.gamma * PRIORITY_3
		elif priority == 0:
			raise ValueError('priority 0 constraints should not have '
							 'slack variables or associated slack '
							 'penalties (gamma)')
		else:
			raise ValueError('argument "priority" must be one of: '
							 '{1, 2, 3}')

	def init_problem(self, n_beams, **options):
		""" Prototype for problem initialization. """
		raise NotImplementedError(
				'solver method "init_problem" not implemented')

	def clear(self):
		""" Prototype for solver tear-down. """
		raise NotImplementedError(
				'solver method "clear" not implemented')

	def __check_dimensions(self, structures):
		"""
		Verify structures have consistently sized dose matrices.

		Arguments:
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects.

		Returns:
			:obj:`int`: Number of beams in treatment plan, corresponds
			to number of columns in dose matrix for each structure
			(should be equal across all structures).

		Raises:
			ValueError: If any :class:`~conrad.medicine.Structure` in
				``structures`` is lacking dose matrix (or mean dose
				matrix) data, or if the dose matrices have inconsistent
				numbers of columns.
		"""
		cols = [0] * len(structures)
		for i, s in enumerate(structures):
			if s.A is not None:
				cols[i] = s.A.shape[1]
			elif s.A_mean is not None:
				cols[i] = s.A_mean.size
			else:
				raise ValueError('structure {} does not have a dose '
								 'matrix or mean dose vector assigned'
								 ''.format(s.name))
		if sum(np.diff(cols) != 0) > 0:
			raise ValueError('all structures in plan must have a dose '
							 'matrix with same number of beams:\n'
							 'either M x N dose matrix or N x 1 mean '
							 'dose vector, with N = # beams\nSizes: {}'
							 ''.format(cols))
		return cols[0]

	def __construction_report(self, structures):
		"""
		Document how structure data converted to an optimization.

		Arguments:
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects.

		Returns:
			:obj:`str`: String documenting how data in ``structures``
			are parsed to form an optimization problem.
		"""
		report = []
		for structure in structures:
			A = structure.A
			if A is not None:
				matrix_info = str('using dose matrix, dimensions {}x{}'.format(
							  *structure.A.shape))
			if structure.is_target:
				reason  = 'structure is target'
			else:
				if structure.collapsable:
					A = structure.A_mean
					matrix_info = str('using mean dose, dimensions '
									  '1x{}'.format(structure.A_mean.size))
					reason = str('structure does NOT have '
								 'min/max/percentile dose constraints '
								 'OR nonlinear objective')
				else:
					reason = str('structure has min/max/percentile '
								 'dose constraints OR nonlinear '
								 'objective')

			report.append(str('structure {} (label = {}): '
							  '{} (reason: {})'.format(structure.name,
							  structure.label, matrix_info, reason)))
		return report

	def __set_scaling(self, structures):
		weight_scaling, dose_scaling = ObjectiveMethods.apply_joint_scaling(
				structures)
		self.__global_dose_scaling = dose_scaling
		self.__global_weight_scaling = weight_scaling

	def build(self, structures, exact=False, **options):
		""" Prototype for problem construction. """
		raise NotImplementedError(
				'solver method "build" not implemented')

	def get_slack_value(Self, constraint_id):
		""" Prototype for querying slack variable values. """
		raise NotImplementedError(
				'solver method "get_slack_value" not implemented')

	def get_dvh_slope(self, constraint_id):
		"""
		Prototype for querying slopes of restricted percentile constraints.
		"""
		raise NotImplementedError(
				'solver method "get_dvh_slope" not implemented')