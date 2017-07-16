"""
Define classes used to record solver inputs/outputs and maintain a
treatment planning history.
"""
"""
Copyright 2016 Baris Ungun, Anqi Fu

This file is part of CONRAD.

CONRAD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CONRAD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CONRAD.  If not, see <http://www.gnu.org/licenses/>.
"""
from conrad.compat import *

import numpy as np

class RunProfile(object):
	"""
	Record of solver input associated with a treatment planning run.

	Attributes:
		use_slack (:obj:`bool`): ``True`` if solver allowed to construct
			convex problem with slack variables for each dose constraint.
		use_2pass (:obj:`bool`): ``True`` if solver requested to
			construct and solve two problems, one incorporating convex
			restrictions of all percentile-type dose constraints, and a
			second problem formulating exact constraints based on the
			feasible output of the first solver run.
		objectives (:obj:`dict`): Dictionary of objective data
			associated with each structure in plan, keyed by structure
			labels.
		constraints (:obj:`dict`): Dictionary of constraint data
			for each dose constraint on each structure in plan, keyed
			by constraint ID.
		gamma: Master scaling applied to slack penalty term in objective
			when dose constraint slacks allowed.
	"""

	def __init__(self, structures=None, use_slack=True, use_2pass=False,
				 gamma='default'):
		"""
		Initialize and populate a `RunProfile`.

		Arguments:
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects supplied to
				solver.
			use_slack (:obj:`bool`, optional): ``True`` if request to solver
				allowed slacks on dose constraints.
			use_2pass (:obj:`bool`, optional): ``True`` if two-pass planning with
				exact dose constraints requested of solver.
			gamma (optional): Slack penalty scaling supplied to solver.
		"""

		self.use_slack = use_slack
		self.use_2pass = use_2pass

		# list of structure labels
		self.label_order = []

		# dictionary of sizes, keyed by structure label
		self.representation_sizes = {}

		# dictionary of objectives, keyed by structure label
		self.objectives = {}

		# dictionary of constraints, keyed by ID
		self.constraints = {}

		# weight used for slack minimization objective
		self.gamma = gamma

		if structures is not None:
			self.pull_objectives(structures)
			self.pull_constraints(structures)

	def pull_objectives(self, structures):
		"""
		Extract and store dictionaries of objective data from ``structures``.

		Arguments:
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects.

		Returns:
			None
		"""
		for _, s in enumerate(structures):
			obj = s.objective
			self.label_order.append(s.label)
			self.objectives[s.label] = {
				'label' : s.label,
				'name' : s.name,
				'objective' : obj.dict if obj is not None else None
			}
			self.representation_sizes[s.label] = 1 if s.collapsable else s.size

	def pull_constraints(self, structures):
		"""
		Extract and store dictionaries of constraint data from ``structures``.

		Arguments:
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects.

		Returns:
			None
		"""
		if isinstance(structures, dict):
			structures = structures.values()
		for s in structures:
			for cid in s.constraints:
				self.constraints[cid] = {
					'label' : s.label,
					'constraint_id' : cid,
					'constraint' : str(s.constraints[cid])
				}

class RunOutput(object):
	"""
	Record of solver outputs associated with a treatment planning run.

	Attributes:
		optimal_variables (:obj:`dict`): Dictionary of optimal variables
			returned by solver. At a minimum, has entries for the beam
			intensity vectors for the first-pass and second-pass solver
			runs. May include entries for:

				- x (beam intensities),
				- y (voxel doses),
				- mu (dual variable for constraint x>= 0), and
				- nu (dual variable for constraint Ax == y).

		optimal_dvh_slopes (:obj:`dict`): Dictionary of optimal slopes
			associated with the convex restriction of each
			percentile-type dose constraint. Keyed by constraint ID.
		solver_info (:obj:`dict`): Dictionary of solver information. At
			a minimum, has entries solver
			run time (first pass/restricted constraints, and second
			pass/exact constraints).
		"""
	def __init__(self):
		""" Intialize empty `RunOutput`. """

		self.optimal_variables = {'x': None, 'x_exact': None}
		self.optimal_dvh_slopes = {}
		self.optimal_slacks = {}
		self.solver_info = {'time': np.nan, 'time_exact': np.nan}
		self.feasible = False

	@property
	def x(self):
		""" Optimal beam intensities from first-pass solve. """
		return self.optimal_variables['x']

	@property
	def x_exact(self):
		""" Optimal beam intensities from second-pass solve. """
		return self.optimal_variables['x_exact']

	@property
	def solvetime(self):
		""" Run time for first-pass solve (restricted dose constraints). """
		return self.solver_info['time']

	@property
	def solvetime_exact(self):
		""" Run time for second-pass solve (exact dose constraints). """
		return self.solver_info['time_exact']


class RunRecord(object):
	"""
	Attributes:
		profile (:class:`RunProfile`): Record of the objective weights,
			dose constraints, and relevant solver options passed to the
			convex solver prior to planning.
		output (:class:`RunOutput`): Output from the solver, including
			optimal beam intensities, i.e., the treatment plan.
		plotting_data (:obj:`dict`): Dictionary of plotting data from
			case, with entries corresponding to the first (and
			potentially only) plan formed by the solver, as well as
			the exact-constraint version of the same plan, if the
			two-pass planning method was invoked.
	"""

	def __init__(self, structures=None, use_slack=True, use_2pass=False,
				 gamma='default'):
		"""
		Initialize :class:`RunRecord`.

		Pass optional arguments to build :attr:`RunRecord.profile`.
		Initialize (but do not populate) :attr:`RunRecord.output` and
		:attr:`RunRecord.plotting_data`.

		Arguments:
			structures: Iterable collection of
				:class:`~conrad.medicine.Structure` objects.
			use_slack (:obj:`bool`, optional): ``True`` if request to
				solver allowed slacks on dose constraints.
			use_2pass (:obj:`bool`, optional): ``True`` if two-pass
				planning with exact dose constraints requested of
				solver.
			gamma (optional): Slack penalty scaling supplied to solver.
		"""
		self.profile = RunProfile(
				structures=structures,
				use_slack=use_slack,
				use_2pass=use_2pass,
				gamma=gamma)
		self.output = RunOutput()
		self.plotting_data = {0: None, 'exact': None}

	@property
	def feasible(self):
		""" Solver feasibility flag from solver output. """
		return self.output.feasible

	@property
	def info(self):
		""" Solver information from solver output. """
		return self.output.solver_info

	@property
	def x(self):
		""" Optimal beam intensitites from first-pass solution. """
		return self.output.x

	@property
	def x_exact(self):
		""" Optimal beam intensitites from second-pass solution. """
		return self.output.x_exact

	@property
	def x_pass1(self):
		""" Alias for :attr:`RunRecord.x`. """
		return self.x

	@property
	def x_pass2(self):
		""" Alias for :attr:`RunRecord.x_exact`. """
		return self.x_exact

	@property
	def nonzero_beam_count(self, tol=1e-6):
		""" Number of active beams in first-pass solution. """
		if self.x is None:
			raise ValueError('no beam data assigned')
		return np.sum(self.x > tol)

	@property
	def nonzero_beam_count_exact(self, tol=1e-6):
		""" Number of active beams in second-pass solution. """
		if self.x_exact is None:
			raise ValueError(
					'no beam data assigned for exact solution '
					'intensities')
		return np.sum(self.x_exact > tol)

	@property
	def solvetime(self):
		"""
		Run time for first-pass solve (restricted dose constraints).
		"""
		return self.output.solvetime

	@property
	def solvetime_exact(self):
		""" Run time for second-pass solve (exact dose constraints). """
		return self.output.solvetime

class PlanningHistory(object):
	"""
	Class for tracking treatment plans generated by a :class:`~conrad.Case`.

	Attributes:
		runs (:obj:`list` of :class:`RunRecord`): List of treatment
			plans in history, in chronological order.
		run_tags (:obj:`dict`): Dictionary mapping tags of named plans
			to their respective indices in :attr:`PlanningHistory.runs`
	"""

	def __init__(self):
		""" Initialize bare history with no treatment plans. """
		self.runs = []
		self.run_tags = {}

	def __getitem__(self, key):
		"""
		Overload operator [].

		Allow slicing syntax for plan retrieval.

		Arguments:
			key: Key corresponding to a tagged treatment plan, or index
				of a plan in the history's list of plans.

		Returns:
			:class:`RunRecord`: Record of solver inputs and outputs from
			requested treatment planning run.

		Raises:
			ValueError: If ``key`` is neither the key to a tagged run
				nor a positive integer than or equal to the number of
				plans in the history.

		"""
		if key in self.run_tags:
				return self.runs[self.run_tags[key]]
		elif isinstance(key, int):
			if key >= len(self.runs):
				raise ValueError('cannot retrieve (base-0) enumerated '
								 'run "{}" since only {} runs have '
								 'been performed'.format(key, len(self.runs)))
			else:
				return self.runs[key]
		else:
			raise ValueError('key "{}" does not correspond to a tagged '
							 'or enumerated run in this {}'
							 ''.format(key, PlanningHistory))

	def __iadd__(self, other):
		"""
		Overload operator +=.

		Extend case history by appending ``other`` to
		:attr:`PlanningHistory.runs`.

		Arguments:
			other (:class:`RunRecord`): Treatment plan to append to
				history.

		Returns:
			Updated :class:`PlanningHistory` object.

		Raises:
			TypeError: If ``other`` not of type :class:`RunRecord`.

		"""
		if isinstance(other, RunRecord):
			self.runs.append(other)
			return self
		else:
			TypeError('operator += only defined for '
				'rvalues of type conrad.RunRecord')

	def add_run(self, run, tag=None):
		self += run
		if tag is not None:
			self.tag_last(tag)

	def no_run_check(self, property_name):
		"""
		Test whether history includes any treatment plans.

		Helper method for property getter methods.

		Arguments:
			property_name (:obj:`str`): Name to use in error message if
				exception raised.

		Returns:
			None

		Raises:
			ValueError: If no treatment plans exist in history,
				i.e., :attr:`PlanningHistory.runs` has length zero.
		"""
		if len(self.runs) == 0:
			raise ValueError(
					'no optimization runs performed, cannot retrieve '
					'{} for most recent plan'.format(property_name))

	@property
	def last_feasible(self):
		""" Solver feasibility flag from most recent treatment plan. """
		self.no_run_check('solver feasibility')
		return self.runs[-1].feasible

	@property
	def last_info(self):
		""" Solver info from most recent treatment plan. """
		self.no_run_check('solver info')
		return self.runs[-1].info

	@property
	def last_x(self):
		""" Vector of beam intensities from most recent treatment plan. """
		self.no_run_check('beam intensitites')
		return self.runs[-1].x

	@property
	def last_x_exact(self):
		""" Second-pass beam intensities from most recent treatment plan. """
		self.no_run_check('beam intensities')
		return self.runs[-1].x_exact

	@property
	def last_solvetime(self):
		""" Solver runtime from most recent treatment plan. """
		self.no_run_check('solve time')
		return self.runs[-1].solvetime

	@property
	def last_solvetime_exact(self):
		""" Second-pass solver runtime from most recent treatment plan. """
		self.no_run_check('solve time')
		return self.runs[-1].solvetime_exact

	def tag_last(self, tag):
		"""
		Tag most recent treatment plan in history.

		Arguments:
			tag: Name to apply to most recently added treatment plan.
				Plan can then be retrieved with slicing syntax::

					# (history is a :class:`PlanningHistory` instance)
					history[tag]

		Returns:
			None

		Raises:
			ValueError: If no treatment plans exist in history.
		"""
		if len(self.runs) == 0:
			raise ValueError(
					'no optimization runs performed, cannot apply tag '
					'"{}" to most recent plan'.format(tag))
		self.run_tags[tag] = len(self.runs) - 1