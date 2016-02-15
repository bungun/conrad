from warnings import warn

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

		# dictionary of constraints, keyed by ID
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
			for cid, dc in s.constraints.iteritems():
				self.constraints[cid] = {'label' : label,
				'constraint_id' : cid, 'constraint' : str(dc) }


class RunOutput(object):
	""" TODO: docstring """
	def __init__(self):
		""" TODO: docstring """

		# x (beams), y (dose)
		# lambda (dual variable for x>= 0)
		self.optimal_variables = {'x': None, 'x_exact': None}
		self.optimal_dvh_slopes = {}
		self.solver_info = {}
		self.feasible = False

	@property
	def x(self):
		return self.optimal_variables['x']

	@property
	def x_exact(self):
		return self.optimal_variables['x_exact']


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


class PlanningHistory(object):
	def __init__(self):
		self.runs = []
		self.run_tags = {}

	def __iadd__(self, other):
		if isinstance(other, RunRecord):
			self.runs.append(other)
			return self
		else:
			TypeError('operator += only defined for '
				'rvalues of type conrad.RunRecord')


	@property
	def last_feasible(self):
		if len(self.runs) == 0: return False
		return self.runs[-1].output.feasible

	@property
	def last_info(self):
		if len(self.runs) == 0: return None
		return self.runs[-1].output.solver_info

	@property
	def last_x(self):
		if len(self.runs) == 0: return None
		return self.runs[-1].output.x


	@property
	def last_x_exact(self):
		if len(self.runs) == 0: return None
		return self.runs[-1].output.x_exact

	def tag_last(self, tag):
		if len(self.runs) == 0:
			warn(Warning('no runs to tag'))
			return
		self.run_tags[tag] = len(self.runs) - 1

		
	def __getitem__(self, runID = None):
		if len(self.runs) == 0:
			warn(Warning('no runs exist in history, returning "None"'))
			return None
		if runID in self.run_tags:
			return self.runs[self.run_tags[runID]]
		elif runID > 0 and runID <= self.run_count:
			return self.runs[runID]				
		elif run is None or run > self.run_count:
			return self.runs[-1]
		else:
			warn(Warning('invalid run number / run tag used in request ',
				'returning "None"'))
			return None