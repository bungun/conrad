from numpy import nan

from conrad.compat import *
from conrad.defs import module_installed
from conrad.optimization.solver_base import *

if module_installed('optkit'):
	import optkit as ok

	class SolverOptkit(Solver):
		""" TODO: docstring """

		def __init__(self):
			""" TODO: docstring """
			Solver.__init__(self)
			self.objective_voxels = None
			self.objective_beams = None
			self.pogs_solver = None
			self.__A_current = None

		# methods:
		def init_problem(self, n_beams, use_slack=True, use_2pass=False,
						 **options):
			""" TODO: docstring """
			self.__n_beams = int(n_beams)

			# self.use_slack = use_slack
			# self.use_2pass = use_2pass
			# self.__x = Variable(n_beams)
			# self.objective = Minimize(0)
			# self.constraints = [self.__x >= 0]
			# self.dvh_vars = {}
			# self.slack_vars = {}
			# self.problem = Problem(self.objective, self.constraints)
			# self.gamma = options.pop('gamma', GAMMA_DEFAULT)

		@property
		def n_beams(self):
			return self.__n_beams

		@property
		def can_solve(self, structures):
			return all([s.constraints.size == 0 for s in structures])

		def clear(self):
			""" TODO: docstring """
			if self.pogs_solver:
				del self.pogs_solver
				ok.backend.reset_device()
				self.pogs_solver = None

		def build(self, structures, exact=False):
			if not self.can_solve:
				raise ValueError(
						'SolverOptkit does not support dose constraints')

			A, dose, weight_abs, weight_lin = \
					self.gather_matrix_and_coefficients(structures)

			if self.__A_current:
				matrix_updated = (self.__A_current != A).sum() > 0
			else:
				matrix_updated = True

			self.__A_current = A
			n_voxels, n_beams = A.shape

			rebuild_f = bool(self.objective_voxels is None or
						   self.objective_voxels.size != n_voxels)
			rebuild_g = bool(self.objective_beams is None or
						   self.objective_beams.size != n_beams)

			if rebuild_f:
				# f(y) = weight_abs|y - dose\ + weight_lin(y - dose)
				self.objective_voxels = ok.PogsObjective(
						n_voxels, h='Abs', b=dose, c=weight_abs, d=weight_lin)
			else:
				self.objective_voxels.set(b=dose, c=weight_abs, d=weight_lin)

			# g(x) = Ind { x >= 0 }
			if rebuild_g:
				self.objective_beams = ok.PogsObjective(n_beams, h='IndGe0')

			if self.pogs_solver is None or matrix_updated:
				self.pogs_solver = ok.PogsSolver(A)

			return self._Solver__construction_report(structures)

		@staticmethod
		def __percentile_constraint_restricted(A, x, constr, beta, slack = None):
			""" Form the upper (or lower) DVH constraint:

				upper constraint:

					\sum (beta + (Ax - (b + slack)))_+ <= beta * vox_limit

				lower constraint:

					\sum (beta - (Ax - (b - slack)))_+ <= beta * vox_limit

			"""
			raise ValueError('dose constraints not supported for SolverOptkit')

		@staticmethod
		def __percentile_constraint_exact(A, x, y, constr, had_slack = False):
			# """ TODO: docstring """
			raise ValueError('dose constraints not supported for SolverOptkit')


		def __add_constraints(self, structure, exact=False):
			# """ TODO: docstring """
			raise ValueError('dose constraints not supported for SolverOptkit')


		def get_slack_value(self, constr_id):
			return nan
			# if constr_id in self.slack_vars:
			# 	return self.slack_vars[constr_id].value
			# else:
			# 	return None

		def get_dual_value(self, constr_id):
			return nan
			# if constr_id in self.__constraint_indices:
			# 	return self.problem.constraints[
			# 			self.__constraint_indices[constr_id]].dual_value[0]
			# else:
			# 	return None

		def get_dvh_slope(self, constr_id):
			return nan
			# beta = self.dvh_vars[constr_id].value if constr_id in self.dvh_vars else None
			# return 1. / beta if beta is not None else None

		@property
		def x(self):
			return self.pogs_solver.output.x

		@property
		def x_dual(self):
			return self.pogs_solver.output.mu

		@property
		def y_dual(self):
			return self.pogs_solver.output.nu

		@property
		def solvetime(self):
			# TODO: time run
		    return self.pogs_solver.info.c.solve_time

		@property
		def status(self):
			return self.pogs_solver.info.err

		@property
		def objective_value(self):
			return self.pogs_solver.info.objval

		@property
		def solveiters(self):
		    return self.pogs_solver.info.iters

		def solve(self, **options):
			self.pogs_solver.solve(self.objective_voxels, self.objective_beams,
								   **options)
			return self.pogs_solver.info.converged

else:
	SolverOptkit = lambda: None
