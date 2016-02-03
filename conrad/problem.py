class PlanningProblem(object):
	def __init__(self, **options):
		pass
	# members:
	# objective
	# constraints
	# problem
	# use_2pass, use_slack
	#
	# methods:
	def add_term(self, structure):
		# extract clinical objective from structure
		pass

	def add_dvh_constraint(self, structure):
		# extract dvh constraint from structure,
		# make slack variable (if self.use_slack), add
		# slack to self.objective and slack >= 0 to constraints
		pass

	# some methods pertaining to the 2pass method

	def solve(self):
		pass

