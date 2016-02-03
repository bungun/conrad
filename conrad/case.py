"""
case.py docstring
"""

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
		"""Case.__init__() docstring"""
		self.problem = None
		self.clinical_spec = None
		self.voxel_labels = None
		self.structures = {}

		# dose matrix
		self.A = None
		# self.shape = (self.voxels, self.beams) = A.shape

		# (most recent) beam intensity design
		self.x = None
		self.run_records = {}

	def add_dvh_constraint(self):
		pass
		# return constr id?

	def drop_dvh_constraint(self, constr_id):
		pass

	def form_objective(self):
		# do something with self.structures, self.problem
		pass

	def form_constraints(self):
		pass

	def form_problem(self):
		pass

	def plan(self):
		# should emit run record with run id
		pass

	def get_design(self):
		pass



