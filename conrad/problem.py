from cvxpy import *
from numpy import inf, array, squeeze, ones, copy as np_copy
from conrad.dvh import DVHCurve, DoseConstraint
from conrad.dose import Constraint, PercentileConstraint, MinConstraint, MaxConstraint, MeanConstraint
from time import clock

GAMMA_DEFAULT = 1e-2
RELTOL_DEFAULT = 1e-3
ABSTOL_DEFAULT = 1e-4
VERBOSE_DEFAULT = 1
MAXITER_DEFAULT = 2000

PRIORITY_1 = 9
PRIORITY_2 = 4
PRIORITY_3 = 1


def println(*msg):
	print(msg)

# TODO: unit test
"""
TODO: problem.py docstring
"""

class Solver(object):
	def __init__(self):
		self.use_2pass = False
		self.use_slack = True
		self.__x = None
		self.__gamma = GAMMA_DEFAULT
		self.dvh_vars = {}
		self.slack_vars = {}
		self.dvh_indices = {}
		self.feasible = False

	@property
	def gamma(self):
		return self.__gamma

	@gamma.setter
	def gamma(self, gamma):
		self.__gamma = gamma if gamma is not None else GAMMA_DEFAULT

	@staticmethod
	def get_cd_from_wts(wt_over, wt_under):
		""" TODO: docstring """
		alpha = wt_over / wt_under
		c = (alpha + 1) / 2
		d = (alpha + 1) / 2
		return c, d

	def gamma_prioritized(self, priority):
		if priority == 1:
			return self.gamma * PRIORITY_1
		elif priority == 2:
			return self.gamma * PRIORITY_2
		elif priority == 3:
			return self.gamma * PRIORITY_1
		else:
			Exception('priority 0 constraints should '
				'not have slack or associated slack penalties')

	def init_problem(self, n_beams, **options):
		pass

	def clear(self):
		pass

	def add_term(self, structure):
		pass

	def add_constraints(self, structure):
		pass

	def get_slack_value(Self, constraint_id):
		pass

	def get_dvh_slope(self, constraint_id):
		pass

class SolverCVXPY(Solver):
	""" TODO: docstring """

	def __init__(self):
		""" TODO: docstring """
		Solver.__init__(self)
		self.objective = None
		self.constraints = None
		self.problem = None

	# methods:
	def init_problem(self, n_beams, **options):
		""" TODO: docstring """
		self.use_slack = options.pop('dvh_no_slack', True)
		self.use_2pass = options.pop('dvh_2pass', False)
		self.__x = Variable(n_beams)
		self.objective = Minimize(0)
		self.constraints = [self.__x >= 0]
		self.dvh_vars = {}
		self.slack_vars = {}
		self.problem = Problem(self.objective, self.constraints)
		self.gamma = options.pop('gamma', GAMMA_DEFAULT)

	def clear(self):
		""" TODO: docstring """
		self.constraints = [self.__x >= 0]
		self.objective = Minimize(0)
		self.problem = Problem(self.objective, self.constraints)

	def add_term(self, structure):
		"""extract clinical objective from structure"""
		A = structure.A
		b = structure.dose
		x = self.__x
		if structure.is_target:
			c, d = self.get_cd_from_wts(structure.w_over, structure.w_under)
			self.problem.objective += Minimize(c * sum_entries(abs(A * x - b)) + d * sum_entries((A * x - b)))
		else:
			if structure.collapsable: A = structure.A_mean
			self.problem.objective += Minimize(structure.w_over * sum_entries(A * x))

	@staticmethod
	def __percentile_constraint_restriction(A, x, constr, beta, slack = None):
		""" Form the upper (or lower) DVH constraint: 

			upper constraint: \sum (beta + (Ax - (b + slack)))_+ <= beta * vox_limit
			lower constraint: \sum (beta - (Ax - (b - slack)))_+ <= beta * vox_limit

		"""
		if not isinstance(constr, PercentileConstraint):
			TypeError('parameter constr must be of type '
				'conrad.dose.PercentileConstraint. '
				'Provided: {}'.format(type(constr)))

		sign = 1 if constr.upper else -1
		fraction = constr.percentile / 100. if constr.upper else 1. - constr.percentile / 100.
		p = fraction * A.shape[0]
		b = constr.dose
		if slack is None: slack = 0.
		return sum_entries(pos( beta + sign * (A * x - (b + sign * slack)) )) <= beta * p

	@staticmethod
	def __percentile_constraint_exact(A, x, y, constr, had_slack = False):
		""" TODO: docstring """
		if not isinstance(constr, DoseConstraint):
			TypeError('parameter constr must be of type '
				'conrad.dose.PercentileConstraint. '
				'Provided: {}'.format(type(constr)))

		sign = 1 if constr.upper else -1
		b = constr.dose_achieved if had_slack else constr.dose
		idx_exact = constr.get_maxmargin_fulfillers(y, had_slack)
		A_exact = np_copy(A[idx_exact, :])
		return sign * (A_exact * x - b) <= 0

	def add_constraints(self, structure, exact = False):
		""" TODO: docstring """
		# extract dvh constraint from structure,
		# make slack variable (if self.use_slack), add
		# slack to self.objective and slack >= 0 to constraints
		exact &= self.use_2pass and structure.y is not None
		no_slack = not self.use_slack

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
						A, self.__x, structure.y, 
						c, had_slack = self.use_slack)

					# add it to problem
					self.problem.constraints += [ dvh_constr ]

				else:
					# beta = 1 / slope for DVH constraint approximation
					beta = Variable(1)
					self.dvh_vars[cid] = beta
					self.problem.constraints += [ beta >= 0 ]

					# build convex restriction to constraint
					dvh_constr = self.__percentile_constraint_restriction(
						structure.A, self.__x, c, beta, slack)

					# add it to problem
					self.problem.constraints += [ dvh_constr ] 

	
	def get_slack_value(self, constr_id):
		return self.slack_vars[constr_id].value if constr_id in self.slack_vars else None

	def get_dvh_slope(self, constr_id):
		beta = self.dvh_vars[constr_id].value if constr_id in self.dvh_vars else None
		return 1. / beta if beta is not None else None

	@property
	def x(self):
		return squeeze(array(self.__x.value))

	@property
	def x_dual(self):
		try:
			return squeee(array(self.problem.constraints[0].dual_value))
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
		solver = options.pop('solver', ECOS)
		reltol = options.pop('reltol', RELTOL_DEFAULT)
		maxiter = options.pop('maxiter', MAXITER_DEFAULT)

		# solve
		PRINT("running solver...")
		if solver == ECOS:
			ret = self.problem.solve(
				solver = ECOS, 
				verbose = VERBOSE, 
				max_iters = maxiter, 
				reltol = reltol,
				reltol_inacc = reltol, 
				feastol = reltol, 
				feastol_inacc = reltol)
		elif solver == SCS:
			ret = self.problem.solve(
				solver = SCS,
				verbose = VERBOSE,
				max_iters = maxiter,
				eps = reltol,
				use_indirect=False)
		else:
			Exception('invalid solver specified: {}\n'
				'no optimization performed'.format(solver))
			return False

		PRINT("status: {}".format(self.problem.status))
		PRINT("optimal value: {}".format(self.problem.value))

		return ret != inf and not isinstance(ret, str)


class PlanningProblem(object):
	""" TODO: docstring """

	def __init__(self):
		""" TODO: docstring """
		self.solver = SolverCVXPY()
		self.use_slack = None
		self.use_2pass = None

	def __update_dvh_constraint(self, structure):
		""" TODO: docstring """
		for cid in structure.constraints:
			slack_var = self.solver.slack_vars[cid]
			slack = 0 if slack_var is None else slack_var.value
			structure.constraints[cid].slack = slack

	def __update_structure(self, structure, exact = False):
		""" TODO: docstring """
		structure.calc_y(self.solver.x)
		if not exact:
			self.__update_dvh_constraint(structure)

	def __gather_solver_info(self, run_output, exact = False):
		keymod = '_exact' if exact else ''
		run_output.solver_info['status' + keymod] = self.solver.status
		run_output.solver_info['time' + keymod] = self.solver.solvetime
		run_output.solver_info['objective' + keymod] = self.solver.objective_value
		run_output.solver_info['iters' + keymod] = self.solver.solveiters

	def __gather_solver_vars(self, run_output, exact = False):
		keymod = '_exact' if exact else ''
		run_output.optimal_variables['x' + keymod] = self.solver.x
		run_output.optimal_variables['lambda' + keymod] = self.solver.x_dual

	def __gather_dvh_slopes(self, run_output, structure_dict):
		# recover dvh constraint slope variables
		for s in structure_dict.itervalues():
			for cid in s.constraints:
				run_output.optimal_dvh_slopes[cid] = self.solver.get_dvh_slope(cid)


	def solve(self, structure_dict, run_output, **options):
		""" TODO: docstring """

		# get number of beams from dose matrix 
		# (attached to any structure)
		# -------------------------------------
		for s in structure_dict.itervalues():
			n_beams = s.A.shape[1]
			break

		# initialize problem with size and options
		# ----------------------------------------
		self.use_slack = options.pop('dvh_slack', True)
		self.use_2pass = options.pop('dvh_exact', False)
		self.solver.init_problem(n_beams, **options)

		# add terms and constraints
		# -------------------------
		for s in structure_dict.itervalues():
			self.solver.add_term(s)
			self.solver.add_constraints(s)

		# solve
		# -----
		start = clock()
		run_output.feasible = self.solver.solve(**options)
		runtime = clock() - start


		# relay output to run_output object
		# ---------------------------------
		self.__gather_solver_info(run_output)
		self.__gather_solver_vars(run_output)
		self.__gather_dvh_slopes(run_output, structure_dict)
		run_output.solver_info['time'] = runtime

		if not run_output.feasible:
			return

		# relay output to structures
		# --------------------------
		for s in structure_dict.itervalues():
			self.__update_structure(s)


		# second pass, if applicable
		# --------------------------
		if self.use_2pass and run_output.feasible:

			self.solver.clear()

			for s in structure_dict.itervalues():
				self.solver.add_term(s)
				self.solver.add_constraints(s, exact = True)

			start = clock()
			self.solver.solve(**options)
			runtime = clock() - start

			self.__gather_solver_info(run_output, exact = True)
			self.__gather_solver_vars(run_output, exact = True)
			run_output.solver_info['time_exact'] = runtime

			for s in structure_dict.itervalues():
				self.__update_structure(s, exact = True)