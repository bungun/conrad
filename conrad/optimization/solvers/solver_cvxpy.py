"""
Define solver using the :mod:`cvxpy` module, if available.

For np.information on :mod:`cvxpy`, see:
http://www.cvxpy.org/en/latest/

If :func:`conrad.defs.module_installed` routine does not find the module
:mod:`cvxpy`, the variable ``SolverCVXPY`` is still defined in this
module's namespace as a lambda returning ``None`` with the same method
signature as the initializer for :class:`SolverCVXPY`. If :mod:`cvxpy`
is found, the class is defined normally.

Attributes:
	SOLVER_DEFAULT (:obj:`str`): Default solver, set to 'SCS' if module
		:mod:`scs` is installed, otherwise set to 'ECOS'.
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

import time
import numpy as np

from conrad.defs import vec as conrad_vec, module_installed, println
from conrad.medicine.dose import Constraint, MeanConstraint, MinConstraint, \
								 MaxConstraint, PercentileConstraint
from conrad.medicine.anatomy import Anatomy
from conrad.optimization.preprocessing import ObjectiveMethods
from conrad.optimization.solvers.environment import *
from conrad.optimization.solvers.solver_base import *

if CVXPY_INSTALLED:
	import cvxpy

	if SCS_INSTALLED:
		SOLVER_DEFAULT = cvxpy.SCS
	else:
		SOLVER_DEFAULT = cvxpy.ECOS

	class SolverCVXPY(Solver):
		"""
		Interface between :mod:`conrad` and :mod:`cvxpy` optimization library.

		:class:`SolverCVXPY` interprets :mod:`conrad` treatment planning
		problems (based on structures with attached objectives, dose
		constraints, and dose matrices) to build equivalent convex
		optimization problems using :mod:`cvxpy`'s syntax.

		The class provides an interface to modify, run, and retrieve
		solutions from optimization problems that can be executed on
		a CPU (or GPU, if :mod:`scs` installed with appropriate backend
		libraries).

		Attributes:
			problem (:class:`cvxpy.Minimize`): CVXPY representation of
				optimization problem.
			constraint_dual_vars (:obj:`dict`): Dictionary, keyed by
				constraint ID, of dual variables associated with each
				dose constraint in the CVXPY problem representation.
				The dual variables' values are stored here after each
				optimization run for access by clients of the
				:class:`SolverCVXPY` object.
		"""

		def __init__(self, n_beams=None, **options):
			"""
			Initialize empty :class:`SolverCVXPY` as :class:`Solver`.

			If number of beams provided, initialize the problem's
			:mod:`cvxpy` representation.

			Arguments:
				n_beams (:obj:`int`, optional): Number of beams in plan.
				**options: Arbitrary keyword arguments, passed to
					:meth:`SolverCVXPY.init_problem`.
			"""
			Solver.__init__(self)
			self.problem = None
			self.__x = cvxpy.Variable(0)
			self.__constraint_indices = {}
			self.constraint_dual_vars = {}
			self.__solvetime = np.nan

			if isinstance(n_beams, int):
				self.init_problem(n_beams, **options)

		def init_problem(self, n_beams, use_slack=True, use_2pass=False,
						 **options):
			"""
			Initialize :mod:`cvxpy` variables and problem components.

			Create a :class:`cvxpy.Variable` of length-``n_beams`` to
			representthe beam  intensities. Invoke
			:meth:`SolverCVXPY.clear` to build minimal problem.

			Arguments:
				n_beams (:obj:`int`): Number of candidate beams in plan.
				use_slack (:obj:`bool`, optional): If ``True``, next
					invocation of :meth:`SolverCVXPY.build` will build
					dose constraints with slack variables.
				use_2pass (:obj:`bool`, optional): If ``True``, next
					invocation of :meth:`SolverCVXPY.build` will build
					percentile-type dose constraints as exact
					constraints instead of convex restrictions thereof,
					assuming other requirements are met.
				**options: Arbitrary keyword arguments.

			Returns:
				None
			"""
			self.__x = cvxpy.Variable(n_beams)
			self.clear()

			self.use_slack = use_slack
			self.use_2pass = use_2pass
			self.gamma = options.pop('gamma', GAMMA_DEFAULT)

		@property
		def n_beams(self):
			""" Number of candidate beams in treatment plan. """
			return self.__x.size[0]

		def clear(self):
			r"""
			Reset :mod:`cvxpy` problem to minimal representation.

			The minmal representation consists of:
				- An empty objective (Minimize 0),
				- A nonnegativity constraint on the vector of beam intensities (:math:`x \ge 0`).

			Reset dictionaries of:
				- Slack variables (all dose constraints),
				- Dual variables (all dose constraints), and
				- Slope variables for convex restrictions (percentile dose constraints).
			"""
			self.problem = cvxpy.Problem(cvxpy.Minimize(0), [self.__x >= 0])
			self.dvh_vars = {}
			self.slack_vars = {}
			self.constraint_dual_vars = {}

		@staticmethod
		def __percentile_constraint_restricted(A, x, constr, beta, slack=None):
			r"""
			Form convex restriction to DVH constraint.

			Upper constraint:

			:math: \sum (beta + (Ax - d + s)))_+ \le \beta * \phi

			Lower constraint::

			:math: \sum (beta - (Ax - d - s)))_+ \le \beta * \phi

			.. math:

			   \mbox{Here, $d$ is a target dose, $s$ is a nonnegative
			   slack variable, and $\phi$ is a voxel limit based on the
			   structure size and the constraint's percentile
			   threshold.}

			Arguments:
				A: Structure-specific dose matrix to use in constraint.
				x (:class:`cvxpy.Variable`): Beam intensity variable.
				constr (:class:`PercentileConstraint`): Dose constraint.
				slack (:obj:`bool`, optional): If ``True``, include
					slack variable in constraint formulation.

			Returns:
				:class:`cvxpy.Constraint`: :mod:`cvxpy` representation
				of convex restriction to dose constraint.

			Raises:
				TypeError: If ``constr`` not of type
					:class:`PercentileConstraint`.
			"""
			if not isinstance(constr, PercentileConstraint):
				raise TypeError('parameter constr must be of type {}'
								'Provided: {}'
								''.format(PercentileConstraint, type(constr)))

			sign = 1 if constr.upper else -1
			fraction = float(sign < 0) + sign * constr.percentile.fraction
			p = fraction * A.shape[0]
			dose = constr.dose.value
			if slack is None:
				slack = 0.
			return cvxpy.sum_entries(cvxpy.pos(
					beta + sign * (A*x - (dose + sign * slack)) )) <= beta * p

		@staticmethod
		def __percentile_constraint_exact(A, x, y, constr, had_slack=False):
			"""
			Form exact version of DVH constraint.

			Arguments:
				A: Structure-specific dose matrix to use in constraint.
				x (:class:`cvxpy.Variable`): Beam intensity variable.
				y: Vector of doses, feasible with respect to constraint
					``constr``.
				constr (:class:`PercentileConstraint`): Dose constraint.
				slack (:obj:`bool`, optional): If ``True``, include
					slack variable in constraint formulation.

			Returns:
				:class:`cvxpy.Constraint`: :mod:`cvxpy` representation
				of exact dose constraint.

			Raises:
				TypeError: If `constr` not of type `PercentileConstraint`.
			"""
			if not isinstance(constr, PercentileConstraint):
				raise TypeError('parameter constr must be of type {}. '
								'Provided: {}'
								''.format(Constraint, type(constr)))

			sign = 1 if constr.upper else -1
			dose = constr.dose_achieved if had_slack else constr.dose
			idx_exact = constr.get_maxmargin_fulfillers(y, had_slack)
			A_exact = np.copy(A[idx_exact, :])
			return sign * (A_exact * x - dose.value) <= 0

		def __add_constraints(self, structure, exact=False):
			"""
			Add constraints from ``structure`` to problem.

			Constraints built with slack variables if
			:attr:`SolverCVXPY.use_slack` is ``True`` at call time. When
			slacks are used, each slack variable is registered in the
			dictionary :attr:`SolverCVXPY.slack_vars` under the
			corresponding constraint's ID as a key for later retrieval
			of optimal values.

			A nonnegativity constraint on each slack variable is
			added to :attr:`SolverCVXPY.problem.constraints`.

			When ``structure`` includes percentile-type dose constraints,
			and a convex restriction (i.e., a hinge loss approximation)
			is used, the reciprocal of the slope of the hinge loss is a
			variable; each such slope variable is registered in the
			dictionary :attr:`SolverCVXPY.dvh_vars` under the
			corresponding constraint's ID as a key for later retrival of
			optimal values.

			A nonnegativity constraint on each slope variable is added
			to :attr:`SolverCVXPY.problem.constraints`.

			Arguments:
				structure (:class:`~conrad.medicine.Structure`):
					Structure from which to read dose matrix and dose
					constraints.
				exact (:obj:`bool`, optional): If ``True`` *and*
					:attr:`SolverCVXPY.use_2pass` is ``True`` *and*
					``structure`` has a calculated dose vector, treat
					percentile-type dose constraints as exact
					constraints. Otherwise, use convex restrictions.

			Returns:
				None

			Raises:
				ValueError: If ``exact`` is ``True``, but other
					conditions for building exact constraints not met.
			"""
			# extract dvh constraint from structure,
			# make slack variable (if self.use_slack), add
			# slack to self.objective and slack >= 0 to constraints
			if exact:
				if not self.use_2pass or structure.y is None:
					raise ValueError('exact constraints requested, but '
									 'cannot be built.\nrequirements:\n'
									 '-input flag "use_2pass" must be '
									 '"True" (provided: {})\n-structure'
									 ' dose must be calculated\n'
									 '(structure dose: {})\n'
									 ''.format(self.use_2pass, structure.y))

			for cid in structure.constraints:
				c = structure.constraints[cid]
				cslack = not exact and self.use_slack and c.priority > 0
				if cslack:
					gamma = self.gamma_prioritized(c.priority)
					slack = cvxpy.Variable(1)
					self.slack_vars[cid] = slack
					self.problem.objective += cvxpy.Minimize(gamma * slack)
					self.problem.constraints += [slack >= 0]
					if not c.upper:
						self.problem.constraints += [slack <= c.dose.value]
				else:
					slack = 0.
					self.slack_vars[cid] = None

				if isinstance(c, MeanConstraint):
					if c.upper:
						self.problem.constraints += [
								structure.A_mean * self.__x - slack <=
								c.dose.value]
					else:
						self.problem.constraints += [
								structure.A_mean * self.__x + slack >=
								c.dose.value]

				elif isinstance(c, MinConstraint):
					self.problem.constraints += \
						[structure.A * self.__x >= c.dose.value]

				elif isinstance(c, MaxConstraint):
					self.problem.constraints += \
						[structure.A * self.__x <= c.dose.value]

				elif isinstance(c, PercentileConstraint):
					if exact:
						# build exact constraint
						dvh_constr = self.__percentile_constraint_exact(
								structure.A, self.__x, structure.y, c,
								had_slack=self.use_slack)

						# add it to problem
						self.problem.constraints += [ dvh_constr ]

					else:
						# beta = 1 / slope for DVH constraint approximation
						beta = cvxpy.Variable(1)
						self.dvh_vars[cid] = beta
						self.problem.constraints += [ beta >= 0 ]

						# build convex restriction to constraint
						dvh_constr = self.__percentile_constraint_restricted(
							structure.A, self.__x, c, beta, slack)

						# add it to problem
						self.problem.constraints += [ dvh_constr ]

				self.__constraint_indices[cid] = None
				self.__constraint_indices[cid] = len(self.problem.constraints) - 1

		def get_slack_value(self, constr_id):
			"""
			Retrieve slack variable for queried constraint.

			Arguments:
				constr_id (:obj:`str`): ID of queried constraint.

			Returns:
				``None`` if ``constr_id`` does not correspond to a
				registered slack variable. ``0`` if corresponding
				constraint built without slack. Value of slack variable
				if constraint built with slack.
			"""
			if constr_id in self.slack_vars:
				if self.slack_vars[constr_id] is None:
					return 0.
				else:
					return self.slack_vars[constr_id].value
			else:
				return None

		def get_dual_value(self, constr_id):
			"""
			Retrieve dual variable for queried constraint.

			Arguments:
				constr_id (:obj:`str`): ID of queried constraint.

			Returns:
				``None`` if ``constr_id`` does not correspond to a
				registered dual variable. Value of dual variable
				otherwise.
			"""
			if constr_id in self.__constraint_indices:
				dual_var = self.problem.constraints[
						self.__constraint_indices[constr_id]].dual_value
				if dual_var is not None:
					if not isinstance(dual_var, float):
						return conrad_vec(dual_var)
					else:
						return dual_var
			else:
				return None

		def get_dvh_slope(self, constr_id):
			"""
			Retrieve slope variable for queried constraint.

			Arguments:
				constr_id (:obj:`str`): ID of queried constraint.

			Returns:
				``None`` if ``constr_id`` does not correspond to a
				registered slope variable. 'NaN' (as :attr:`numpy.np.nan`)
				if constraint built as exact. Reciprocal of slope
				variable otherwise.
			"""
			if constr_id in self.dvh_vars:
				beta =  self.dvh_vars[constr_id].value
				if beta is None:
					return np.nan
				return 1. / beta
			else:
				return None

		@property
		def x(self):
			""" Vector variable of beam intensities, x. """
			return conrad_vec(self.__x.value)

		@property
		def x_dual(self):
			""" Dual variable corresponding to constraint x >= 0. """
			try:
				return conrad_vec(self.problem.constraints[0].dual_value)
			except:
				return None

		@property
		def solvetime(self):
			""" Solver run time. """
			return self.__solvetime

		@property
		def setuptime(self):
			""" Solver run time. """
			return np.nan

		@property
		def status(self):
			""" Solver status. """
			return self.problem.status

		@property
		def objective_value(self):
			""" Objective value at end of solve. """
			return self.problem.value

		@property
		def solveiters(self):
			""" Number of solver iterations performed. """
			return 'n/a'

		def __objective_expression(self, structure):
			structure.normalize_objective()
			if structure.collapsable:
				return structure.objective.expr(structure.A_mean.T * self.__x)
			else:
				return structure.objective.expr(
						structure.A * self.x, structure.voxel_weights)

		def build(self, structures, exact=False, **options):
			"""
			Update :mod:`cvxpy` optimization based on structure data.

			Extract dose matrix, target doses, and objective weights
			from structures.

			Use doses and weights to add minimization terms to
			:attr:`SolverCVXPY.problem.objective`. Use dose constraints
			to extend :attr:`SolverCVXPY.problem.constraints`.

			(When constraints include slack variables, a penalty on each
			slack variable is added to the objective.)

			Arguments:
				structures: Iterable collection of :class:`Structure`
					objects.

			Returns:
				:obj:`str`: String documenting how data in
				``structures`` were parsed to form an optimization
				problem.
			"""
			self.clear()
			if isinstance(structures, Anatomy):
				structures = structures.list
			# A, dose, weight_abs, weight_lin = \
					# self._Solver__gather_matrix_and_coefficients(structures)

			self.problem.objective = cvxpy.Minimize(0)
			for s in structures:
				self.problem.objective += cvxpy.Minimize(
						ObjectiveMethods.expr(s, self.__x))
				self.__add_constraints(s, exact=exact)

			# self.problem.objective = cvxpy.Minimize(
			# 		weight_abs.T * cvxpy.abs(A * self.__x - dose) +
			# 		weight_lin.T * (A * self.__x - dose))

			# for s in structures:
				# self.__add_constraints(s, exact=exact)

			return self._Solver__construction_report(structures)

		def solve(self, **options):
			"""
			Execute optimization of a previously built planning problem.

			Arguments:
				**options: Keyword arguments specifying solver options,
					passed to :meth:`cvxpy.Problem.solve`.

			Returns:
				:obj:`bool`: ``True`` if :mod:`cvxpy` solver converged.

			Raises:
				ValueError: If specified solver is neither 'SCS' nor
					'ECOS'.
			"""

			# set verbosity level
			VERBOSE = bool(options.pop('verbose', VERBOSE_DEFAULT))
			PRINT = println if VERBOSE else lambda msg : None

			# solver options
			solver = options.pop('solver', SOLVER_DEFAULT)
			reltol = float(options.pop('reltol', RELTOL_DEFAULT))
			maxiter = int(options.pop('maxiter', MAXITER_DEFAULT))
			use_gpu = bool(options.pop('gpu', GPU_DEFAULT))
			use_indirect = bool(options.pop('use_indirect', INDIRECT_DEFAULT))

			# solve
			PRINT('running solver...')
			start = time.clock()
			if solver == cvxpy.ECOS:
				ret = self.problem.solve(
						solver=cvxpy.ECOS,
						verbose=VERBOSE,
						max_iters=maxiter,
						reltol=reltol,
						reltol_inacc=reltol,
						feastol=reltol,
						feastol_inacc=reltol)
			elif solver == cvxpy.SCS:
				if use_gpu:
					ret = self.problem.solve(
							solver=cvxpy.SCS,
							verbose=VERBOSE,
							max_iters=maxiter,
							eps=reltol,
							gpu=use_gpu)
				else:
					ret = self.problem.solve(
							solver=cvxpy.SCS,
							verbose=VERBOSE,
							max_iters=maxiter,
							eps=reltol,
							use_indirect=use_indirect)
			else:
				raise ValueError('invalid solver specified: {}\n'
								 'no optimization performed'.format(solver))
			self.__solvetime = time.clock() - start


			PRINT("status: {}".format(self.problem.status))
			PRINT("optimal value: {}".format(self.problem.value))

			return ret != np.inf and not isinstance(ret, str)

else:
	SOLVER_DEFAULT = 'CVXPY_UNAVAILABLE'
	SolverCVXPY = lambda: None
