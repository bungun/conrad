from conrad.dvh import DoseConstraint
from conrad.structure import Structure
from conrad.problem import PlanningProblem
from conrad.run_data import RunRecord
# TODO: imports
# TODO: unit test
"""
TODO: case.py docstring
"""

def gen_constraint_id(label, constraint_count):
	""" TODO: docstring """
	return "sid:{}:cid:{}".format(label, constraint_count)

def constraint2label(constr_id):
	""" TODO: docstring """
	return constr_id.split(:)[1]

def default_weights(is_target = False):
	""" TODO: docstring """
	if is_target:
		# w_under = 1, w_over = 0.05
		return 1., 0.05
	else:
		return None, 0.1


class Case(object):
	"""TODO:

	initialize a case with:
		- dose matrix, A, matrix \in R^m x n, m = # voxels, n = # beams
		- voxel labels, (int) vector in \R^m
		- prescription: dictionary where each element has the fields:
			label (int, also the key)
			dose (float)
			is_target (bool)
			dose_constraints (dict, or probably string/tuple list)
	"""

	def __init__(self):
		"""TODO: Case.__init__() docstring"""
		self.problem = PlanningProblem()
		self.prescription = None
		self.voxel_labels = None
		self.structures = {}

		# counts go with life of Case object,
		# even if, e.g., all constraints removed from a run
		self.constraint_count = 0
		self.run_count = 0

		# for now, assume A sorted!
		# need: structure sizes, structure pointers
		# need: structure order (plotting)
		# need: parse full mat + data into Structure()s
		# need: digest clinical spec
		# some kind of plotting!

		# dose matrix
		self.A = None
		# self.shape = (self.voxels, self.beams) = A.shape

		self.run_records = {}

	def add_dvh_constraint(self, label, dose, fraction, direction):
		""" TODO: docstring """
		self.constraint_count += 1
		constr = DoseConstraint(dose, fraction, direction)
		cid = gen_constraint_id(label, self.constraint_count)
		self.structures[label].add_constraint(cid, constr)
		return cid

	def drop_dvh_constraint(self, constr_id):
		""" TODO: docstring """
		label = constraint2label(constr_id)
		self.structures[label].remove_constraint(constr_id)

	def change_objective(self, label, dose = None, 
		w_under = None, w_over = None):
		""" TODO: docstring """
		self.structures[label].set_objective(dose, w_under, w_over)

	def plan(self, *args, **kwargs):
		""" TODO: docstring """

		# check for targets
		if not self.has_targets:
			print str("Warning: plan has no targets."
				"Not running optimization.\n\n")
			return

		# use 2 pass OFF by default
		use_2pass = 'dvh_2pass' in args

		# dvh slack ON by default
		use_slack = not 'dvh_no_slack' in args

		# objective weight for slack minimization
		gamma = kwargs['dvh_wt_slack'] if 'dvh_wt_slack' in kwargs else None
		if gamma is not None: kwargs['gamma'] = gamma

		rr = RunRecord(self.structures, 
			use_2pass = use_2pass, 
			use_slack = use_slack, 
			gamma = gamma)


		# solve problem
		self.problem.solve(self.structures, rr.output, *args, **kwargs)

		self.run_count += 1
		self.run_records[run_count] = rr


	def calc_doses(self, x):
		""" TODO: docstring """
		for s in self.structures.itervals():
			s.calc_y(x)

	@property
	def plotting_data(self):
		""" TODO: docstring """
		d = {}
		for label, s in self.structures.iteritems():
			d[label] = s.plotting_data
	    return d
		
	@property
	def n_structures(self):
		""" TODO: docstring """
	    return len(self.structures.keys())
	
	@property
	def n_beams(self):
		""" TODO: docstring """
		return self.A.shape[1]

	@property
	def n_dvh_constraints(self):
		""" TODO: docstring """
		return sum([dc.count for dc in self.dvh_constrs_by_struct])
	
	@property
	def has_targets(self):
		""" TODO: docstring """
		for s in self.structures.iteritems():
			if s.is_target: return True
		return False

