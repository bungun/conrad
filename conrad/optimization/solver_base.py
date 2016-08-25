"""
Define base class for interfaces between CONRAD and Python modules
with convex solvers, e.g., `cvxpy`.

Attributes:
	GAMMA_DEFAULT (float): Default scaling to apply to objective term
		penalizing weighted sum of slack variables, when dose constraint
		slacks are allowed.
	RELTOL_DEFAULT (float): Default relative tolerance for solver.
	ABSTOL_DEFAULT (float): Default absolute tolerance for solver.
	VERBOSE_DEFAULT (int): Default solver verbosity.
	MAXITER_DEFAULT (int): Default maximum solver iterations.
	INDIRECT_DEFAULT (bool): Default solver mode (applies to SCS only).
	GPU_DEFAULT (bool): Default solver device (applies to SCS, POGS).
	PRIORITY_1 (int): Penalty scaling for slack on high priority
		dose constraints.
	PRIORITY_2 (int): Penalty scaling for slack on medium priority
		dose constraints.
	PRIORITY_3 (int): Penalty scaling for slack on low priority
		dose constraints.

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
from numpy import diff, zeros
from conrad.compat import *

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

class Solver(object):
	"""
	Base class for translating CONRAD planning requests to convex problems.

	Attributes:
		use_2pass (bool): When True, enables exact percentile-type dose
			constraints to be built (other conditions must hold).
		use_slack (bool): When True, dose constraints are built with
			slack.
		dvh_vars (:obj:`dict`): Dictionary, keyed by constraint ID, of
			values for slope variables associated with the convex
			restriction of each percentile-type dose constraint in the
			problem.
		slack_vars (:obj:`dict`): Dictionary, keyed by constraint ID, of
			values for slack variables associated with each dose
			constraint in the problem.
		feasible (bool): True if most recent optimization run was
			feasible.
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

	@property
	def gamma(self):
		""" Scaling to apply to weighted sum of slack variables. """
		return self.__gamma

	@gamma.setter
	def gamma(self, gamma):
		if gamma:
			self.__gamma = float(gamma)

	@staticmethod
	def get_cd_from_wts(wt_under, wt_over):
		"""
		Convert piecewise linear weights to absolute value + affine.

		Given an objective function f_i: R -> R that consists of
		separate affine penalties for underdosing and overdosing::
			# f_i = w_+ (y_i - dose_i) + w_i (y_i - dose_i)

		rephrase as mixture of absolute value plus affine terms::
			# f_i = c |y_i - dose_i| + d (y_i - dose_i)

		Arguments:
			wt_under (float): Underdose weight, value should be
				nonnegative.
			wt_over (float): Overdose weight, value should be positive.

		Returns:
			:obj:`tuple` of float: Weights for absolute value and affine
				terms that result in an equivalent objective function.
		"""
		c = (wt_over + wt_under) / 2.
		d = (wt_over - wt_under) / 2.
		return c, d

	def gamma_prioritized(self, priority):
		"""
		Calculate penalty scaling for slack variable.

		Arguments:
			priority (int): Constraint priority, should be 1, 2 or 3.

		Returns:
			float: Constraint-specific penalty obtained by multiplying
				master scaling for slack variable penalties
				(`Solver.gamma`) times priority-based scaling.

		Raises:
			ValueError: If `priority` is not one of 1, 2 or 3.

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
		raise RuntimeError('solver method "init_problem" not implemented')

	def clear(self):
		""" Prototype for solver tear-down. """
		raise RuntimeError('solver method "clear" not implemented')

	def __check_dimensions(self, structures):
		"""
		Verify structures have consistently sized dose matrices.

		Arguments:
			structures: Iterable collection of
				`conrad.medicine.Structure` objects.

		Returns:
			int: Number of beams in treatment plan, corresponds to
				number of columns in dose matrix for each structure
				(should be equal across all structures).

		Raises:
			ValueError: If any `conrad.medicine.Structure` in
				`structures` is lacking dose matrix (or mean dose
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
		if sum(diff(cols) != 0) > 0:
			raise ValueError('all structures in plan must have a dose '
							 'matrix with same number of beams:\n'
							 'either M x N dose matrix or N x 1 mean '
							 'dose vector, with N = # beams\nSizes: {}'
							 ''.format(cols))
		return cols[0]

	def __gather_matrix_and_coefficients(self, structures):
		"""
		Gather dose matrix and objective parameters from structures.

		The objective to be built is of the form::
			# w_abs^T * |Ax - dose| + w_lin * (Ax - dose)

		Procedure for gathering dose matrix::
			# Set A = [] empty matrix with 0 rows and N columns.
			#
			# for each structure in structures do
			#	if structure is collapsable (mean/no dose constraints):
			#		append structure's 1 x N mean dose vector to A.
			#	else:
			#		append structure's M_structure x N dose matrix to A.
			# end for

		The dose vector is built by repeating the structure dose for as
		many rows as the structure's dose data occupies in the coalesced
		dose matrix (once for collapsed structure, M_structure times for
		full structures). These subvectors are concatenated vertically.

		The weight vectors are built by converting the underdose and
		overdose penalties for each structure into equivalent absolute
		value and affine penalties, and multiplying each by the
		structure's vector of voxel weights (or by the sum of the
		structure's voxel weights, if the structure is collapsable).
		These subvectors are concatenated vertically.

		Arguments:
			structures: Iterable collection of
				`conrad.medicine.Structure` objects.

		Returns:
			:obj:`tuple`: Tuple of dose matrix, target dose vector,
				absolute value penalty weight vector and affine penalty
				weight vector.
		"""
		cols = self.__check_dimensions(structures)
		rows = sum([s.size if not s.collapsable else 1 for s in structures])
		A = zeros((rows, cols))
		dose = zeros(rows)
		weight_abs = zeros(rows)
		weight_lin = zeros(rows)
		ptr = 0

		for s in structures:
			if s.collapsable:
				A[ptr, :] = s.A_mean[:]
				weight_abs[ptr] = s.w_over * sum(s.voxel_weights)
				weight_lin[ptr] = 0
				ptr += 1
			else:
				A[ptr : ptr + s.size, :] += s.A_full
				if s.is_target:
					c_, d_ = self.get_cd_from_wts(s.w_under, s.w_over)
					dose[ptr : ptr + s.size] = s.dose.value
					weight_abs[ptr : ptr + s.size] = c_ * s.voxel_weights
					weight_lin[ptr : ptr + s.size] = d_ * s.voxel_weights
				else:
					dose[ptr : ptr + s.size] = 0
					weight_abs[ptr : ptr + s.size] = s.w_over * s.voxel_weights
					weight_lin[ptr : ptr + s.size] = 0
				ptr += s.size

		return A, dose, weight_abs, weight_lin

	def __construction_report(self, structures):
		"""
		Document how structure data converted to an optimization.

		Arguments:
			structures: Iterable collection of `Structure` objects.

		Returns:
			:obj:`str`: String documenting how data in `structures`
				are parsed to form an optimization problem.
		"""
		report = []
		for structure in structures:
			A = structure.A
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
								 'min/max/percentile dose constraints')
				else:
					reason = str('structure has min/max/percentile '
								 'dose constraints')

			report.append(str('structure {} (label = {}): '
							  '{} (reason: {})'.format(structure.name,
							  structure.label, matrix_info, reason)))
		return report

	def build(self, structures, exact=False):
		""" Prototype for problem construction. """
		raise RuntimeError('solver method "build" not implemented')

	def get_slack_value(Self, constraint_id):
		""" Prototype for querying slack variable values. """
		raise RuntimeError('solver method "get_slack_value" not implemented')

	def get_dvh_slope(self, constraint_id):
		"""
		Prototype for querying slopes of restricted percentile constraints.
		"""
		raise RuntimeError('solver method "get_dvh_slope" not implemented')