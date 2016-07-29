from time import clock
from os import getenv
from numpy import inf, array, squeeze, ones, zeros, copy as np_copy
from cvxpy import *

from conrad.compat import *
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
		self.solver = None
		self.use_slack = None
		self.use_2pass = None

	def __update_constraints(self, structure):
		""" TODO: docstring """

		# STUB: for now, avoid doing anything when using pogs solver

		for cid in structure.constraints:
			slack_var = self.solver.get_slack_value(cid)
			slack = 0 if slack_var is None else slack_var.value
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
		run_output.optimal_variables['lambda' + keymod] = self.solver.x_dual

	def __gather_dvh_slopes(self, run_output, structures):
		# recover dvh constraint slope variables
		for s in structures:
			for cid in s.constraints:
				run_output.optimal_dvh_slopes[cid] = \
						self.solver.get_dvh_slope(cid)

	def __set_solver_fastest_available(self, structures):
		if self.solver_pogs:
			if self.solver_pogs.can_solve(structures):
				self.solver = self.solver_pogs
				return
		elif self.solver_cvxpy:
			self.solver = self.solver_cvxpy
		else:
			raise Exception('no solvers available')

	def solve(self, structures, run_output, **options):
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
		self.use_slack = options.pop('dvh_slack', True)
		self.use_2pass = options.pop('dvh_exact', False)
		self.__set_solver_fastest_available(structures)
		self.solver.init_problem(n_beams, use_slack=self.use_slack,
								 use_2pass=self.use_2pass, **options)

		# build problem
		construction_report = self.solver.build(structures)

		if PRINT_PROBLEM_CONSTRUCTION:
			print('\nPROBLEM CONSTRUCTION:')
			for cr in construction_report:
				print(cr)

		# solve
		start = clock()
		run_output.feasible = self.solver.solve(**options)
		runtime = clock() - start

		# relay output to run_output object
		self.__gather_solver_info(run_output)
		self.__gather_solver_vars(run_output)
		self.__gather_dvh_slopes(run_output, structure_dict)
		run_output.solver_info['time'] = runtime

		if not run_output.feasible:
			return

		# relay output to structures
		for s in structures:
			self.__update_structure(s)

		# second pass, if applicable
		if self.use_2pass and run_output.feasible:

			self.solver.clear()
			self.solver.build(structure_dict.values(), exact=True)

			start = clock()
			self.solver.solve(**options)
			runtime = clock() - start

			self.__gather_solver_info(run_output, exact = True)
			self.__gather_solver_vars(run_output, exact = True)
			run_output.solver_info['time_exact'] = runtime

			for s in structure_dict.values():
				self.__update_structure(s, exact = True)