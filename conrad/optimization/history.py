from warnings import warn
from conrad.compat import *

# TODO: unit test
"""
TODO: run_data.py docstring
"""
class RunProfile(object):
	""" TODO: docstring """
	def __init__(self, structures=None, use_slack=True, use_2pass=False,
				 gamma='default'):
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
		for label, s in structures.items():
			self.objectives[label] = {
				'label' : label,
				'name' : s.name,
				'dose' : s.dose,
				'w_under' : s.w_under_raw,
				'w_over' : s.w_over_raw
			}

	def pull_constraints(self, structures):
		""" TODO: docstring """
		for label, s in structures.iteritems():
			for cid, dc in s.constraints.iteritems():
				self.constraints[cid] = {
					'label' : label,
					'constraint_id' : cid,
					'constraint' : str(dc)
				}

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

	@property
	def solvetime(self):
		return self.solver_info.pop('time', None)

	@property
	def solvetime_exact(self):
		return self.solver_info.pop('time_exact', None)


class RunRecord(object):
	""" TODO: docstring """
	def __init__(self, structures=None, use_slack=True, use_2pass=False,
				 gamma='default'):

		""" TODO: docstring """
		self.profile = RunProfile(
				structures=structures,
				use_slack=use_slack,
				use_2pass=use_2pass,
				gamma=gamma)
		self.output = RunOutput()
		self.plotting_data = {0: None, 'exact': None}

		@property
		def feasible(self):
			return self.output.feasible

		@property
		def info(self):
			return self.output.solver_info

		@property
		def x(self):
			return self.output.x

		@property
		def x_exact(self):
			return self.output.x_exact

		@property
		def nonzero_beam_count(self, tol=1e-6):
			return sum(self.x > tol)

		@property
		def nonzero_beam_count_exact(self, tol=1e-6):
			return sum(self.x_exact > tol)

		@property
		def solvetime(self):
			return self.output.solvetime

		@property
		def solvetime_exact(self):
			return self.output.solvetime

class PlanningHistory(object):
	def __init__(self):
		self.runs = []
		self.run_tags = {}

	def __getitem__(self, key):
		if isinstance(key, int):
			if key >= len(self.runs):
				raise ValueError('cannot retrieve (base-0) enumerated '
								 'run "{}" since only {} runs have '
								 'been performed'.format(key, len(self.runs)))
			else:
				return self.runs[key]
		elif key in self.run_tags:
				return self.runs[self.run_tags[key]]
		else:
			raise ValueError('key "{}" does not correspond to a tagged '
							 'or enumerated run in this {}'
							 ''.format(key, PlanningHistory))

	def __iadd__(self, other):
		if isinstance(other, RunRecord):
			self.runs.append(other)
			return self
		else:
			TypeError('operator += only defined for '
				'rvalues of type conrad.RunRecord')

	def no_run_check(self, property_name):
		if len(self.runs) == 0:
			raise Exception('no optimization runs performed, cannot '
							'retrieve {} for most recent run'
							''.format(property_name))

	@property
	def last_feasible(self):
		self.no_run_check('solver feasibility')
		return self.runs[-1].feasible

	@property
	def last_info(self):
		self.no_run_check('solver info')
		return self.runs[-1].info

	@property
	def last_x(self):
		self.no_run_check('beam intensitites')
		return self.runs[-1].x

	@property
	def last_x_exact(self):
		self.no_run_check('beam intensities')
		return self.runs[-1].x_exact

	@property
	def last_solvetime(self):
		self.no_run_check('solve time')
		return self.runs[-1].solvetime

	@property
	def last_solvetime_exact(self):
		self.no_run_check('solve time')
		return self.runs[-1].solvetime_exact


	def tag_last(self, tag):
		if len(self.runs) == 0:
			raise Exception('no optimization runs performed, cannot '
							'apply tag "{}" to most recent run'.format(tag))
		self.run_tags[tag] = len(self.runs) - 1