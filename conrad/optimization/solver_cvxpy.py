from conrad.compat import *
from conrad.defs import vec, module_installed
from conrad.optimization.solver_base import *

if module_installed('cvxpy'):
	from cvxpy import *

	if module_installed('scs'):
		SOLVER_DEFAULT = SCS
	else:
		SOLVER_DEFAULT = ECOS

	class SolverCVXPY(Solver):
		""" TODO: docstring """

		def __init__(self):
			""" TODO: docstring """
			Solver.__init__(self)
			self.objective = None
			self.constraints = None
			self.problem = None
			self.__x = Variable(0)
			self.__constraint_indices = {}
			self.constraint_dual_vars = {}

		# methods:
		def init_problem(self, n_beams, use_slack=True, use_2pass=False,
						 **options):
			""" TODO: docstring """
			self.use_slack = use_slack
			self.use_2pass = use_2pass
			self.__x = Variable(n_beams)
			self.objective = Minimize(0)
			self.constraints = [self.__x >= 0]
			self.dvh_vars = {}
			self.slack_vars = {}
			self.problem = Problem(self.objective, self.constraints)
			self.gamma = options.pop('gamma', GAMMA_DEFAULT)

		@property
		def n_beams(self):
			return self.__x.size[0]

		def clear(self):
			""" TODO: docstring """
			self.constraints = [self.__x >= 0]
			self.objective = Minimize(0)
			self.problem = Problem(self.objective, self.constraints)

		def build(self, structures, exact=False):
			A, dose, weight_abs, weight_lin = \
					self.gather_matrix_and_coefficients(structures)


			self.problem.objective = Minimize(
					weight_abs.T * abs(A * self.__x - dose) +
					weight_lin.T * (A * self.__x - dose))

			for s in structures:
				self.__add_constraints(s, exact=exact)

			return self._Solver__construction_report(structures)

		@staticmethod
		def __percentile_constraint_restricted(A, x, constr, beta, slack=None):
			""" Form the upper (or lower) DVH constraint:

				upper constraint:

					\sum (beta + (Ax - (b + slack)))_+ <= beta * vox_limit

				lower constraint:

					\sum (beta - (Ax - (b - slack)))_+ <= beta * vox_limit

			"""
			if not isinstance(constr, PercentileConstraint):
				raise TypeError('parameter constr must be of type {}'
								'Provided: {}'
								''.format(PercentileConstraint, type(constr)))

			sign = 1 if constr.upper else -1
			fraction = sign < 0 + sign * constr.percentile.fraction
			p = fraction * A.shape[0]
			b = constr.dose.value
			if slack is None:
				slack = 0.
			return sum_entries(pos(
					beta + sign * (A * x - (b + sign * slack)) )) <= beta * p

		@staticmethod
		def __percentile_constraint_exact(A, x, y, constr, had_slack = False):
			""" TODO: docstring """
			if not isinstance(constr, Constraint):
				raise TypeError('parameter constr must be of type {}. '
								'Provided: {}'
								''.format(Constraint, type(constr)))

			sign = 1 if constr.upper else -1
			b = constr.dose_achieved if had_slack else constr.dose
			idx_exact = constr.get_maxmargin_fulfillers(y, had_slack)
			A_exact = np_copy(A[idx_exact, :])
			return sign * (A_exact * x - b) <= 0

		def __add_constraints(self, structure, exact=False):
			""" TODO: docstring """
			# extract dvh constraint from structure,
			# make slack variable (if self.use_slack), add
			# slack to self.objective and slack >= 0 to constraints
			if exact:
				if not self.use_2pass or structure.y is None:
					raise ValueError('exact constraints requested, but '
									 'cannot be built.\nrequirements:\n'
									 'input flag "use_2pass" must be '
									 '"True" (provided: {})\nstructure '
									 'dose must be calculated\n'
									 '(structure dose: {}\n'
									 ''.format(self.use_2pass, structure.y))

			for cid in structure.constraints:
				c = structure.constraints[cid]
				cslack = not exact and self.use_slack and c.priority > 0
				if cslack:
					gamma = self.gamma_prioritized(c.priority)
					slack = Variable(1)
					self.slack_vars[cid] = slack
					self.problem.objective += Minimize(gamma * slack)
					self.problem.constraints += [slack >= 0]
				else:
					slack = 0.
					self.slack_vars[cid] = None

				if isinstance(c, MeanConstraint):
					if c.upper:
						self.problem.constraints += \
							[structure.A_mean * self.__x - slack <= c.dose]
					else:
						self.problem.constraints += \
							[structure.A_mean * self.__x + slack >= c.dose]

				elif isinstance(c, MinConstraint):
					self.problem.constarints += \
						[structure.A * self.__x >= c.dose]

				elif isinstance(c, MaxConstraint):
					self.problem.constarints += \
						[structure.A * self.x <= c.dose]

				elif isinstance(c, PercentileConstraint):
					if exact:
						# build exact constraint
						dvh_constr = self.__percentile_constraint_exact(
							structure.A, self.__x, structure.y,
							c, had_slack = self.use_slack)

						# add it to problem
						self.problem.constraints += [ dvh_constr ]

					else:
						# beta = 1 / slope for DVH constraint approximation
						beta = Variable(1)
						self.dvh_vars[cid] = beta
						self.problem.constraints += [ beta >= 0 ]

						# build convex restriction to constraint
						dvh_constr = self.__percentile_constraint_restricted(
							structure.A, self.__x, c, beta, slack)

						# add it to problem
						self.problem.constraints += [ dvh_constr ]

		def get_slack_value(self, constr_id):
			if constr_id in self.slack_vars:
				return self.slack_vars[constr_id].value
			else:
				return None

		def get_dual_value(self, constr_id):
			if constr_id in self.__constraint_indices:
				return self.problem.constraints[
						self.__constraint_indices[constr_id]].dual_value[0]
			else:
				return None

		def get_dvh_slope(self, constr_id):
			if constr_id in self.dvh_vars:
				beta =  self.dvh_vars[constr_id].value
				return 1. / beta
			else:
				return None

		@property
		def x(self):
			return vec(self.__x.value)

		@property
		def x_dual(self):
			try:
				return vec(self.problem.constraints[0].dual_value)
			except:
				return None

		@property
		def solvetime(self):
			# TODO: time run
		    return 'n/a'

		@property
		def status(self):
			return self.problem.status

		@property
		def objective_value(self):
			return self.problem.value

		@property
		def solveiters(self):
			# TODO: get solver iters
			return 'n/a'

		def solve(self, **options):
			""" TODO: docstring """

			# set verbosity level
			VERBOSE = bool(options.pop('verbose', VERBOSE_DEFAULT))
			PRINT = println if VERBOSE else lambda : None

			# solver options
			solver = options.pop('solver', SOLVER_DEFAULT)
			reltol = float(options.pop('reltol', RELTOL_DEFAULT))
			maxiter = int(options.pop('maxiter', MAXITER_DEFAULT))
			use_gpu = bool(options.pop('gpu', GPU_DEFAULT))
			use_indirect = bool(options.pop('use_indirect', INDIRECT_DEFAULT))

			# solve
			PRINT('running solver...')
			if solver == ECOS:
				ret = self.problem.solve(
						solver=ECOS,
						verbose=VERBOSE,
						max_iters=maxiter,
						reltol=reltol,
						reltol_inacc=reltol,
						feastol=reltol,
						feastol_inacc=reltol)
			elif solver == SCS:
				if use_gpu:
					ret = self.problem.solve(
							solver=SCS,
							verbose=VERBOSE,
							max_iters=maxiter,
							eps=reltol,
							gpu=use_gpu)
				else:
					ret = self.problem.solve(
							solver=SCS,
							verbose=VERBOSE,
							max_iters=maxiter,
							eps=reltol,
							use_indirect=use_indirect)
			else:
				raise ValueError('invalid solver specified: {}\n'
								 'no optimization performed'.format(solver))

			PRINT("status: {}".format(self.problem.status))
			PRINT("optimal value: {}".format(self.problem.value))

			return ret != inf and not isinstance(ret, str)

else:
	SolverCVXPY = lambda: None
