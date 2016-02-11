# TODO: unit test
"""
TODO: run_data.py docstring
"""
class RunProfile(object):
	""" TODO: docstring """
	def __init__(self, structures = None,
		use_slack = True, 
		use_2pass = False, 
		gamma = 'default'):

		""" TODO: docstring """
		self.use_slack = use_slack
		self.use_2pass = use_2pass

		# dictionary of objective, keyed by structure label
		self.objectives = {}

		# list of constraints
		self.constraints = {}

		# weight used for slack minimization objective
		self.gamma = gamma

		if structures is not None:
			self.pull_objectives(structures)
			self.pull_constraints(structures)


	def pull_objectives(self, structures):
		""" TODO: docstring """
		for label, s in structures.iteritems():
			self.objectives[label] = {'label' : label,
			'name' : s.name, 'dose' : s.dose, 
			'w_under' : s.w_under_raw,
			'w_over' : s.w_over_raw }

	def pull_constraints(self, structures):
		""" TODO: docstring """
		for label, s in structures.iteritems():
			for cid, dc in s.dose_constraints.iteritems():
				self.constraints[cid] = {'label' : label,
				'constraint_id' : cid, 'dose' : dc.dose_requested,
				'percentile' : 100 * dc.fraction,
				'direction' : dc.direction }

	def push_objectives(self, structures):
		""" TODO: docstring """
		for label in structures:
			if not self.objectives.has_key(label):
				print str('RunProfile.objectives dictionary '
					'must have an entry for each key in '
					'the dictionary structures. '
					'Objectives not pushed')
				return

		for label, obj  in self.objectives.iteritems():
			structures[label].set_objective(
				obj['dose'], obj['w_under'], obj['w_over'])


class RunOutput(object):
	""" TODO: docstring """
	def __init__(self):
		""" TODO: docstring """

		# x (beams), y (dose)
		# lambda (dual variable for x>= 0)
		self.optimal_variables = {}
		self.optimal_dvh_slopes = {}
		self.solver_info = {}
		self.feasible = False

class RunRecord(object):
	""" TODO: docstring """
	def __init__(self, structures = None,
		use_slack = True, 
		use_2pass = False, 
		gamma = 'default'):
			
		""" TODO: docstring """
		self.profile = RunProfile(structures = structures,
			use_slack = use_slack, use_2pass = use_2pass,
			gamma = gamma)
		self.output = RunOutput()
