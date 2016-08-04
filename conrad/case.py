from operator import add
from warnings import warn
from numpy import ndarray, array, squeeze, zeros

from conrad.compat import *
from conrad.physics import Physics
from conrad.medicine import Anatomy, Prescription
from conrad.optimization.problem import PlanningProblem
from conrad.optimization.history import RunRecord, PlanningHistory

"""
TODO: case.py docstring
"""

class Case(object):
	"""TODO: docstring

	initialize a case with:
		- dose matrix (A): matrix \in R^m x n, m = # voxels, n = # beams
		- voxel labels: vector{int} in \R^m
		- label_order: list{labels}, (labels are integers) TODO: finish description
		- prescription: dict{labels, dict}, where each element has the fields:
			label (int, also the key)
			dose (float)
			is_target (bool)
			dose_constraints (dict, or probably string/tuple list)
	"""
	def __init__(self, anatomy=None, physics=None, prescription=None,
				 **options):
		self.__physics = None
		self.__anatomy = None
		self.__prescription = None
		self.__problem = None

		self.physics = physics
		self.anatomy = anatomy
		self.prescription = prescription
		self.__problem = PlanningProblem()

		if self.anatomy.is_empty:
			self.anatomy.structures = self.prescription.structure_dict

		# append prescription constraints unless suppressed:
		suppress_rx_constraints = options.pop('suppress_rx_constraints', False)
		if not suppress_rx_constraints:
			self.transfer_constraints_to_anatomy()

	@property
	def physics(self):
		return self.__physics

	@physics.setter
	def physics(self):
		self.__physics = Physics(physics)

	@property
	def anatomy(self):
		return self.__anatomy

	@anatomy.setter
	def anatomy(self, anatomy):
		self.__anatomy = Anatomy(anatomy)

	@property
	def prescription(self):
		return self.__prescription

	@prescription.setter
	def prescription(self, prescription):
		self.__prescription = Prescription(prescription)

	@property
	def problem(self):
		return self.__problem

	@property
	def structures(self):
		return self.anatomy.structures

	@property
	def A(self):
	    return self.physics.dose_matrix

	@property
	def n_structures(self):
		return self.anatomy.n_structures

	@property
	def n_voxels(self):
		return self.physics.voxels

	@property
	def n_beams(self):
		return self.phsyics.beams

	def transfer_constraints_to_anatomy(self):
		constraint_dict = self.prescription.constraints_by_label
		for label, constr_list in constraint_dict.items():
			self.antomy[structure_label].constraints += constr_list

	def add_constraint(self, structure_label, threshold, direction, dose):
		""" TODO: docstring """
		if '<' in direction or '<=' in direction:
			self.anatomy[structure_label].constraints += D(threshold) <= dose
		else:
			self.anatomy[structure_label].constraints += D(threshold) >= dose

		return self.anatomy[structure_label].constraints.last_key

	def drop_constraint(self, constr_id):
		""" TODO: docstring """
		for s in self.anatomy:
			if constr_id in s.constraints:
				s.constraints -= constr_id

	def clear_constraints(self):
		""" TODO: docstring """
		self.anatomy.clear_constraints()

	def change_constraint(self, constr_id, threshold, direction, dose):
		for s in self.anatomy:
			if constr_id in s.constraints:
				s.set_constraint(constr_id, threshold, direction, dose)

	def change_objective(self, label, dose=None, w_under=None, w_over=None):
		""" TODO: docstring """
		if self.structures[label].is_target:
			if dose:
				self.structures[label].dose = dose
			if w_under:
				self.structures[label].w_under = w_under
		if w_over:
			self.structures[label].w_over = w_over

	def load_physics_to_anatomy(self):
		if self.anatomy.plannable and self.physics.data_loaded:
			raise ValueError('case.anatomy already has valid dose '
							 'matrix data loaded to each structure.\n'
							 'call this method with option '
							 '"overwrite=True" to load new dose matrix '
							 'data')
		if self.physics.data_loaded:
			return

		for structure in self.anatomy:
			label = structure.label
			structure.A_full = self.physics.dose_matrix_by_label(label)
			structure.voxel_weights = self.physics.voxel_weights_by_label(
					label)
			self.physics.mark_data_as_loaded()

	def calculate_doses(self, x):
		""" TODO: docstring """
		self.anatomy.calculate_doses(x)

	def plan(self, **options):
		plannable = True
		if not self.physics.plannable:
			plannable &= False
		elif not self.anatomy.plannable:
			self.load_physics_to_anatomy()
			plannable &= self.anatomy.plannable
		if not plannable:
			raise ValueError('case not plannable in current state.\n'
							 'minimum requirements:\n'
							 '---------------------\n'
							 '-"case.physics.dose_matrix" is set\n'
							 '-"case.physics.voxel_labels" is set\n'
							 '-"case.anatomy" contains at least one '
							 'structure marked as target')


		# two pass planning for DVH constraints: OFF by default
		use_2pass = options['dvh_exact'] = options.pop('dvh_exact', False)
		# dose constraint slack: ON by default
		use_slack = options['dvh_slack'] = options.pop('dvh_slack', True)

		# objective weight for slack minimization
		gamma = options['gamma'] = options.pop('slack_penalty', None)

		run = RunRecord(
				self.structures,
				use_2pass=use_2pass,
				use_slack=use_slack,
				gamma=gamma)

		# solve problem
		self.problem.solve(self.anatomy, run.output, **options)

		# update doses
		if run.feasible:
			run.plotting_data[0] = self.plotting_data(run.x)
			if use_2pass:
				run.plotting_data['exact'] = self.plotting_data(run.x_exact)
			return True, run
		else:
			warn('Problem infeasible as formulated')
			return False, run

	@property
	def plotting_data(self, x=None):
		""" TODO: docstring """
		if x:
			self.calculate_doses(x)

 		d = {}
		for structure in self.anatomy:
			d[structure.label] = structure.plotting_data
		return d