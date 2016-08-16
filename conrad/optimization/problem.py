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
from os import getenv
from numpy import inf, array, squeeze, ones, zeros, copy as np_copy

from conrad.compat import *
from conrad.medicine.dose import PercentileConstraint
from conrad.optimization.solver_cvxpy import SolverCVXPY
from conrad.optimization.solver_optkit import SolverOptkit

"""
TODO: problem.py docstring
"""

class PlanningProblem(object):
	""" TODO: docstring """

	def __init__(self):
		""" TODO: docstring """
		self.solver_cvxpy = SolverCVXPY()
		self.solver_pogs = SolverOptkit()
		self.__solver = None

	@property
	def solver(self):
		if self.__solver is None:
			if self.solver_cvxpy is not None:
				return self.solver_cvxpy
			elif self.solver_pogs is not None:
				return self.solver_pogs
			else:
				raise ValueError('no solvers avaialable')
		else:
			return self.__solver

	def __update_constraints(self, structure, slack_tol=1e-3):
		""" TODO: docstring """
		for cid in structure.constraints:
			slack = self.solver.get_slack_value(cid)
			if slack is not None:
				# dead zone between (-slack_tol, +slack_tol) set to 0
				# positive slacks get updated
				# negative slacks get rejected
				if slack < -slack_tol or slack > slack_tol:
					structure.constraints[cid].slack = slack

	def __update_structure(self, structure, exact=False):
		""" TODO: docstring """
		structure.calc_y(self.solver.x)
		if not exact:
			self.__update_constraints(structure)

	def __gather_solver_info(self, run_output, exact=False):
		keymod = '_exact' if exact else ''
		run_output.solver_info['status' + keymod] = self.solver.status
		run_output.solver_info['time' + keymod] = self.solver.solvetime
		run_output.solver_info['objective' + keymod] = self.solver.objective_value
		run_output.solver_info['iters' + keymod] = self.solver.solveiters

	def __gather_solver_vars(self, run_output, exact=False):
		keymod = '_exact' if exact else ''
		run_output.optimal_variables['x' + keymod] = self.solver.x
		run_output.optimal_variables['mu' + keymod] = self.solver.x_dual
		if self.solver == self.solver_pogs:
			run_output.optimal_variables['nu' + keymod] = self.solver.y_dual
		else:
			run_output.optimal_variables['nu' + keymod] = None


	def __gather_dvh_slopes(self, run_output, structures):
		# recover dvh constraint slope variables
		for s in structures:
			for cid in s.constraints:
				run_output.optimal_dvh_slopes[cid] = \
						self.solver.get_dvh_slope(cid)

	def __set_solver_fastest_available(self, structures):
		if self.solver_pogs is not None:
			if self.solver_pogs.can_solve(structures):
				self.__solver = self.solver_pogs
				return
		if self.solver_cvxpy is not None:
			self.__solver = self.solver_cvxpy
			return

		raise Exception('no solvers available')

	def __verify_2pass_applicable(self, structures):
		percentile_constraints_included = False
		for s in structures:
			for key in s.constraints:
				percentile_constraints_included |= isinstance(
						s.constraints[key], PercentileConstraint)
		return percentile_constraints_included

	def solve(self, structures, run_output, slack=True,
			  exact_constraints=False, **options):
		""" TODO: docstring """
		if self.solver_cvxpy is None and self.solver_pogs is None:
			raise Exception('at least one of packages\n-cvxpy\n'
							'-optkit\nmust be installed to perform '
							'optimization')
		if 'print_construction' in options:
			PRINT_PROBLEM_CONSTRUCTION = bool(options['print_construction'])
		else:
			PRINT_PROBLEM_CONSTRUCTION = getenv('CONRAD_PRINT_CONSTRUCTION',
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
		construction_report = self.solver.build(structures)

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
		run_output.solver_info['time'] = self.solver.solvetime

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

			for s in structures:
				self.__update_structure(s, exact=True)

			return 2
		else:
			return 1