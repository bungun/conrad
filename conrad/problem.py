from cvxpy import *
from numpy import inf, array, squeeze
from conrad.dvh import DVHCurve, DoseConstraint

GAMMA_DEFAULT = 1e-2
RELTOL_DEFAULT = 1e-3
ABSTOL_DEFAULT = 1e-4
VERBOSE_DEFAULT = 1
MAXITER_DEFAULT = 2000

def println(*msg):
	print(*msg)

# TODO: unit test
"""
TODO: problem.py docstring
"""

def get_cd_from_wts(wt_over, wt_under):
""" TODO: docstring """
	alpha = wt_over / wt_under
	c = (alpha + 1) / 2
	d = (alpha + 1) / 2
	return c, d

def dvh_constraint_restriction(A, x, constr, beta, slack = None):
	""" Form the upper (or lower) DVH constraint: 

		upper constraint: \sum (beta + (Ax - (b + slack)))_+ <= beta * vox_limit
		lower constraint: \sum (beta - (Ax - (b - slack)))_+ <= beta * vox_limit

	"""
	if not isinstance(constr, DoseConstraint):
		TypeError("parameter constr must be of type "
			"conrad.DoseConstraint. Provided: {}".format(type(constr)))

def dvh_restriction(A, x, b, p, beta, upper = True, slack = 0):
	""" TODO: docstring """
	sign = 1 if constr.upper else -1
	fraction = constr.fraction if constr.upper else 1. - constr.fraction
	p = fraction * A.shape[0]
	b = constr.dose_requested

	if slack is not None:
		return sum_entries(pos( beta + sign * (A * x - (b + sign * slack)) )) <= beta * p
	else:
		return sum_entries(pos( beta + sign * (A * x - b)) ) <= beta * p

def dvh_constraint_exact(A, x, y, constr, had_slack = False):
	""" TODO: docstring """
	if not isinstance(constr, DoseConstraint):
		TypeError("parameter constr must be of type "
			"conrad.DoseConstraint. Provided: {}".format(type(constr)))

	sign = 1 if constr.upper else -1
	if had_slack and constr.dose_actual is not None:
		b = constr.dose_actual
	else:
		b = constr.dose_requested
	idx_exact = constr.get_maxmargin_fulfillers(y)
	A_exact = np.copy(A[idx_exact, :])
	return sign * (A_exact * x - b) <= 0


class PlanningProblem(object):
	""" TODO: docstring """

	def __init__(self):
		""" TODO: docstring """
		self.objective = None
		self.constraints = None
		self.problem = None
		self.use_2pass = False
		self.use_slack = True
		self.x = None
		self.gamma = GAMMA_DEFAULT
		self.dvh_vars = {}
		self.slack_vars = {}
		self.dvh_indices = {}


	# methods:
	def init_problem(self, n_beams, *options, **kwargs):
		""" TODO: docstring """
		self.use_slack = not 'dvh_no_slack' in options
		self.use_2pass = 'dvh_2pass' in options
		self.x = Variable(n_beams)
		self.objective = Minimize(0)
		self.constraints = [self.x >= 0]
		self.dvh_vars = {}
		self.slack_vars = {}
		self.problem = Problem(self.objective, self.constraints)
		if 'gamma' in kwargs:
			self.gamma = kwargs['gamma']


	def clear_dvh_constraints(self):
		""" TODO: docstring """
		self.constraints = [self.x >= 0]
		self.objective = 0
		self.problem = Minimize(self.objective, self.constraints)

	def add_term(self, structure):
		"""extract clinical objective from structure"""
		if structure.is_target:
			A = structure.A
			b = structure.dose
			c, d = get_cd_from_wts(structure.wt_over, structure.wt_under)
			self.problem.objective += Minimize(c.T * abs(A * x - b) + d.T * (A * x - b))

	def add_dvh_constraint(self, structure, exact = False):
		""" TODO: docstring """
		# extract dvh constraint from structure,
		# make slack variable (if self.use_slack), add
		# slack to self.objective and slack >= 0 to constraints
		for cid in structure.dose_constraints:

			if exact and self.use_2pass and structure.y is not None:
				
				# build exact constraint
				dvh_constr = dvh_constraint_exact(structure.A,
					self.x, structure.y, 
					structure.dose_constraints[cid],
					had_slack = self.use_slack)

				# add it to problem
				self.problem.constraints += [ dvh_constr ]

			else:
				# beta = 1 / slope for DVH constraint approximation
				beta = Variable(1)
				self.dvh_vars[cid] = beta
				self.problem.constraints += [ beta >= 0 ]

				if self.use_slack:
					# s = slack in dose for DVH constraint
					s = Variable(1)
					self.problem.objective += Minimize(self.gamma * s)
					self.problem.constraints += [s >= 0]
				else:
					s = None

				# s is cvxpy.Variable, or None
				self.slack_vars[cid] = s

				# build convex restriction to constraint
				dvh_constr = dvh_constraint_restriction(structure.A,
					self.x, structure.dose_constraints[cid], beta, s)

				# add it to problem
				self.problem.constraints += [ dvh_constr ] 


	# some methods pertaining to the 2pass method

	def update_dvh_constraint(self, structure):
		""" TODO: docstring """
		for cid in structure.dose_constraints:
			slack = self.slack_vars[cid].value
			structure.dose_constraints[cid].set_actual_dose(slack)

	def update_structure(self, structure, exact = False):
		""" TODO: docstring """
		structure.calc_y(self.x.value)
		if not exact:
			self.update_dvh_constraint(structure)

	def __solve(self, *options, **kwargs):
		""" TODO: docstring """

		# set verbosity level
		VERBOSE = bool(kwargs['verbose']) if 'verbose' in kwargs else VERBOSE_DEFAULT
		PRINT = println if VERBOSE else lambda : None
		
		# solver options
		solver = SCS if 'SCS' in options else ECOS
		reltol = kwargs['reltol'] if 'reltol' in kwargs else RELTOL_DEFAULT
		maxiter = kwargs['maxiter'] if 'maxiter' in kwargs else MAXITER_DEFAULT

		# solve
		PRINT("running solver...")
		if solver == ECOS:
			ret = self.problem.solve(
				solver = ECOS, 
				verbose = VERBOSE, 
				max_iters = maxiter, 
				reltol = reltol
				reltol_inacc = reltol, 
				feastol = reltol, 
				feastol_inacc = reltol)
		else:
			ret = self.problem.solve(
				solver = SCS,
				verbose = VERBOSE,
				max_iters = maxiter,
				reltol = reltol,
				use_indirect=False)

		PRINT("status: {}".format(prob.status))
		PRINT("optimal value: {}".format(prob.value))

		return ret != inf and not isinstance(ret, str)

	def solve(self, structure_dict, run_output, *options, **kwargs):
		""" TODO: docstring """

		# get number of beams from dose matrix 
		# (attached to any structure)
		# -------------------------------------
		for st in structure_dict.itervalues():
			n_beams = st.A.size[0]
			break

		# initialize problem with size and options
		# ----------------------------------------
		self.init_problem(n_beams, *options, **kwargs)

		# add terms and constraints
		# -------------------------
		for st in structure_dict.itervalues():
			self.add_term(st)
			self.add_dvh_constraint(st)

		# solve
		# -----
		feasible = self.__solve(*options, **kwargs)

		if not feasible:
			print "problem infeasible as formulated"

		# relay output to structures
		# --------------------------
		for st in structure_dict.itervalues():
			self.update_structure(st)

		# relay output to run_output object
		# ---------------------------------

		# recover primal variable x and
		# dual variable lambda associated with ineq. Ax >= 0
		run_output.optimal_variables['x'] = squeeze(array(self.x.value))
		run_output.optimal_variables['lambda'] = squeeze(
			array(self.problem.constraints[0].dual_value))

		# recover dvh constraint slope variables
		for st in structure_dict.itervalues():
			for cid in st.dose_constraints.keys():
				run_output.optimal_dvh_slopes[cid] = 1. / self.dvh_vars[cid].value 

		# second pass, if applicable
		# --------------------------
		if self.use_2pass and feasible:

			self.clear_dvh_constraints()

			for st in structure_dict.itervalues():
				self.add_term(st)
				self.add_dvh_constraint(st, exact = True)	

			self.__solve(*options, **kwargs)

			for st in structure_dict.itervalues():
				self.update_structure(st, exact = True)

			run_output.optimal_variables['x_exact'] = squeeze(
				array(self.x.value))
			run_output.optimal_variables['lambda_exact'] = squeeze(
				array(self.problem.constraints[0].dual_value))