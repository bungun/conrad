"""
Define :class:`PlanningProblem`, interface between :class:`~conrad.Case`
and solvers.
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

import os

from conrad.medicine.dose import PercentileConstraint
from conrad.optimization.solver_cvxpy import SolverCVXPY
from conrad.optimization.solver_optkit import SolverOptkit
from conrad.optimization.history import RunOutput

class PlanningProblem(object):
	"""
	Interface between :class:`~conrad.Case` and convex solvers.

	Builds and solves specified treatment planning problem using fastest
	available solver, then extracts solution data and solver metadata
	(e.g., timing results) for use by clients of the
	:class:`PlanningProblem` object (e.g., a :class:`~conrad.Case`).

	Attributes:
		solver_cvxpy (:class:`SolverCVXPY` or :class:`NoneType`):
			:mod:`cvxpy`-baed solver, if available.
		solver_pogs (:class:`SolverOptkit` or :class:`NoneType`): POGS
			solver, if available.
	"""

	def __init__(self):
		"""
		Initialize a bare :class:`PlanningProblem` with all available solvers.

		Arguments:
			None
		"""
		self.solver_cvxpy = SolverCVXPY()
		self.solver_pogs = SolverOptkit()
		self.__solver = None

	@property
	def solver(self):
		""" Get active solver (CVXPY or OPTKIT/POGS). """
		if self.__solver is None:
			if self.solver_cvxpy is not None:
				return self.solver_cvxpy
			elif self.solver_pogs is not None:
				return self.solver_pogs
			else:
				raise ValueError('no solvers avaialable')
		else:
			return self.__solver

	def __update_constraints(self, structure, slack_tol=5e-3):
		"""
		Pull solver results pertaining to constraint slacks.

		Arguments:
			structure (:class:`~conrad.medicine.Structure`): Structure
				for which to retrieve constraint data.
			slack_tol (:obj:`float`, optional): Numerical tolerance; if
				the magnitude of the slack variable's value is smaller
				than this tolerance, it is treated as zero.

		Returns:
			None

		TODO: retrieve dual variable values.
		"""
		for cid in structure.constraints:
			slack = self.solver.get_slack_value(cid)
			if slack is not None:
				# dead zone between (-slack_tol, +slack_tol) set to 0
				# positive slacks get updated
				# negative slacks get rejected
				if slack < -slack_tol or slack > slack_tol:
					structure.constraints[cid].slack = slack

	def __update_structure(self, structure, exact=False):
		"""
		Calculate structure dose from solver's optimal beam intensities.

		Arguments:
			structure (:class:`~conrad.medicine.Structure`): Structure
				for which to calculate dose.
			exact (:obj:`bool`, optional): If ``False`` (i.e.,
				reading first-pass results), trigger call to update
				constraints as well.

		Returns:
			None
		"""
		structure.calc_y(self.solver.x)
		if not exact:
			self.__update_constraints(structure)

	def __gather_solver_info(self, run_output, exact=False):
		"""
		Transfer solver metadata to a :class:RunOutput` instance.

		Arguments:
			run_output (:class:`RunOutput`):
				Container for solver data. Data stored as dictionary
				entries in :attr:`RunOutput.solver_info`.
			exact (:obj:`bool`, optional): If ``True``, append '_exact'
				to keys of dictionary entries.

		Returns:
			None

		Raises:
			TypeError: If ``run_output`` not of type :class:`RunOutput`.
		"""
		if not isinstance(run_output, RunOutput):
			raise TypeError('Argument "run_output" must be of type {}'
							''.format(RunOutput))
		keymod = '_exact' if exact else ''
		run_output.solver_info['status' + keymod] = self.solver.status
		run_output.solver_info['time' + keymod] = self.solver.solvetime
		run_output.solver_info['objective' + keymod] = self.solver.objective_value
		run_output.solver_info['iters' + keymod] = self.solver.solveiters

	def __gather_solver_vars(self, run_output, exact=False):
		"""
		Transfer solver variables to a :class:`RunOutput` instance.

		Arguments:
			run_output (:class: `RunOutput`): Container for solver data.
				Data stored as dictionary entries in
				:attr:`RunOutput.optimal_variables`.
			exact (:obj:`bool`, optional): If ``True``, append '_exact'
				to keys of dictionary entries.

		Returns:
			None

		Raises:
			TypeError: If ``run_output`` not of type :class:`RunOutput`.
		"""
		if not isinstance(run_output, RunOutput):
			raise TypeError('Argument "run_output" must be of type {}'
							''.format(RunOutput))

		keymod = '_exact' if exact else ''
		run_output.optimal_variables['x' + keymod] = self.solver.x
		run_output.optimal_variables['mu' + keymod] = self.solver.x_dual
		if self.solver == self.solver_pogs:
			run_output.optimal_variables['nu' + keymod] = self.solver.y_dual
		else:
			run_output.optimal_variables['nu' + keymod] = None


	def __gather_dvh_slopes(self, run_output, structures):
		"""
		Transfer optimal DVH constraint slopes to :class:`RunOutput`.

		For each percentile-type dose constraint in each structure in
		``structures``, retrieve the optimal value of the slope variable
		used in the convex restriction to the constraint.

		Arguments:
			run_output (:class:`RunOutput`): Container for solver data.
				Data stored as dictionary entries in
				:attr:`RunOutput.optimal_dvh_slopes`.
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects.

		Returns:
			None

		Raises:
			TypeError: If ``run_output`` not of type :class:`RunOutput`.
		"""
		if not isinstance(run_output, RunOutput):
			raise TypeError('Argument "run_output" must be of type {}'
							''.format(RunOutput))

		# recover dvh constraint slope variables
		for s in structures:
			for cid in s.constraints:
				run_output.optimal_dvh_slopes[cid] = self.solver.get_dvh_slope(
						cid)

	def __gather_constraint_slacks(self, run_output, structures):
		"""
		Transfer optimal dose constraint slacks to :class:`RunOutput`.

		Arguments:
			run_output (:class:`RunOutput`): Container for solver data.
				Data stored as dictionary entries in
				:attr:`RunOutput.optimal_dvh_slopes`.
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects.

		Returns:
			None

		Raises:
			TypeError: If ``run_output`` not of type :class:`RunOutput`.
		"""
		if not isinstance(run_output, RunOutput):
			raise TypeError('Argument "run_output" must be of type {}'
							''.format(RunOutput))

		# recover dvh constraint slope variables
		for s in structures:
			for cid in s.constraints:
				run_output.optimal_slacks[cid] = self.solver.get_slack_value(
						cid)

	def __set_solver_fastest_available(self, structures):
		"""
		Set active solver to fastest solver than can handle problem.

		If ``structures`` includes any dose constraints, only
		:mod:`cvxpy`-based solvers can be used. If no dose constraints
		are present, and the module :mod:`optkit` is installed, the POGS
		solver is the fastest option.

		Arguments:
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects, passed to
				:meth:`SolverOptkit.can_solve`.

		Returns:
			None

		Raises:
			ValueError: If neither a CVXPY solver nor an OPTKIT/POGS
				solver is available.
		"""
		if self.solver_pogs is not None:
			if self.solver_pogs.can_solve(structures):
				self.__solver = self.solver_pogs
				return
		if self.solver_cvxpy is not None:
			self.__solver = self.solver_cvxpy
			return

		raise ValueError('no solvers available')

	def __verify_2pass_applicable(self, structures):
		"""
		Two-pass algorithm only needed if percentile constraints present.

		Arguments:
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects to check for
				percentile constraints.

		Returns:
			:obj:`bool`: ``True`` if any structure in ``structures`` has
			a percentile dose constraint.
		"""
		percentile_constraints_included = False
		for s in structures:
			for key in s.constraints:
				percentile_constraints_included |= isinstance(
						s.constraints[key], PercentileConstraint)
		return percentile_constraints_included

	def solve(self, structures, run_output, slack=True,
			  exact_constraints=False, **options):
		"""
		Run treatment plan optimization.

		Arguments:
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects with
				attached objective, constraint, and dose matrix
				information. Build convex model of treatment planning
				problem using these data.
			run_output (:class:`RunOutput`): Container for saving solver
				results.
			slack (:obj:`bool`, optional): If ``True``, build dose
				constraints with slack.
			exact_constraints (:obj:`bool`, optional): If ``True`` *and*
				at least one structure has a percentile-type dose
				constraint, execute the two-pass planning algorithm,
				using convex restrictions of the percentile constraints
				on the firstpass,  and exact versions of the constraints
				on the second pass.
			**options: Abitrary keyword arguments, passed through to
				:meth:`PlanningProblem.solver.init_problem` and
				:meth:`PlanningProblem.solver.build`.

		Returns:
			:obj:`int`: Number of feasible solver runs performed: ``0``
			if first pass infeasible, ``1`` if first pass feasible,
			``2`` if two-pass method requested and both passes feasible.

		Raises:
			ValueError: If no solvers avaialable.
		"""
		if self.solver_cvxpy is None and self.solver_pogs is None:
			raise ValueError(
					'at least one of packages\n-cvxpy\n-optkit\nmust '
					'be installed to perform optimization')
		if 'print_construction' in options:
			PRINT_PROBLEM_CONSTRUCTION = bool(options['print_construction'])
		else:
			PRINT_PROBLEM_CONSTRUCTION = os.getenv('CONRAD_PRINT_CONSTRUCTION',
												False)

		# get number of beams from dose matrix
		for s in structures:
			n_beams = len(s.A_mean)
			break

		# initialize problem with size and options
		use_slack = options.pop('dvh_slack', slack)
		use_2pass = options.pop('dvh_exact', exact_constraints)
		use_2pass &= self.__verify_2pass_applicable(structures)
		self.__set_solver_fastest_available(structures)
		self.solver.init_problem(n_beams, use_slack=use_slack,
								 use_2pass=use_2pass, **options)

		# build problem
		construction_report = self.solver.build(structures, **options)

		if PRINT_PROBLEM_CONSTRUCTION:
			print('\nPROBLEM CONSTRUCTION:')
			for cr in construction_report:
				print(cr)

		# solve
		run_output.feasible = self.solver.solve(**options)

		# relay output to run_output object
		self.__gather_solver_info(run_output)
		self.__gather_solver_vars(run_output)
		self.__gather_dvh_slopes(run_output, structures)
		self.__gather_constraint_slacks(run_output, structures)
		run_output.solver_info['time'] = self.solver.solvetime
		run_output.solver_info['setup_time'] = self.solver.setuptime

		if not run_output.feasible:
			return 0

		# relay output to structures
		for s in structures:
			self.__update_structure(s)

		# second pass, if applicable
		if use_2pass and run_output.feasible:
			self.solver.build(structures, exact=True)
			self.solver.solve(**options)

			self.__gather_solver_info(run_output, exact=True)
			self.__gather_solver_vars(run_output, exact=True)
			run_output.solver_info['time_exact'] = self.solver.solvetime
			run_output.solver_info['setup_time_exact'] = self.solver.setuptime

			for s in structures:
				self.__update_structure(s, exact=True)

			return 2
		else:
			return 1