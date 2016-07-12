from operator import add
from warnings import warn
from numpy import ndarray, array, squeeze, zeros

from conrad.compat import *
from conrad.physics import Physics
from conrad.medicine import Anatomy, Prescription
from conrad.optimization.problem import PlanningProblem
from conrad.optimization.history import RunRecord

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
	def __init__(self, physics=None, anatomy=None, prescription=None,
				 **options):
		self.__physics = None
		self.__anatomy = None
		self.__prescription = None
		self.__run = None

		if physics:
			self.physics = physics
		if anatomy:
			self.anatomy = anatomy
		if prescription:
			self.prescription = prescription

		self.problem = PlanningProblem()

		# append prescription constraints unless suppressed:
		suppress_rx_constraints = options.pop('suppress_rx_constraints', False)
		if not suppress_rx_constraints:
			self.transfer_constraints_to_anatomy()

	@property
	def physics(self):
		return self.__physics

	@physics.setter
	def physics(self):
		if not isinstance(physics, Physics):
			raise TypeError('argument "physics" must be of type '
							'{}'.format(Physics))
		if physics.dose_matrix is None:
			raise ValueError('argument "physics" must have an attached '
							 'dose matrix')

		self.__physics = physics
		if self.anatomy:
			self.anatomy.import_dose_matrix(self.physics)

	@property
	def anatomy(self):
		return self.__anatomy

	@anatomy.setter
	def anatomy(self, anatomy):
		if not isinstance(anatomy, Anatomy):
			raise TypeError('argument "anatomy" must be of type '
							'{}'.format(Anatomy))
		self.__anatomy = anatomy
		if self.physics:
			self.anatomy.import_dose_matrix(self.physics)


	@property
	def prescription(self):
		return self.__prescription

	@prescription.setter
	def prescription(self, prescription):
		if not isinstance(physics, Prescription):
			raise TypeError('argument "prescription" must be of type '
							'{}'.format(Prescription))

		self.__prescription = prescription

	@property
	def structures(self):
		if self.anatomy:
			return self.anatomy.structures
		else:
			raise AttributeError('cannot retrieve field "structures" '
								 'when case.anatomy is not initialized')

	@property
	def A(self):
	    return self.physics.dose_matrix

	def transfer_constraints_to_anatomy(self):
		constraint_dict = self.prescription.constraints_by_label
		for label, constr_list in constraint_dict.items():
			self.structures[label].constraints += constr_list

	def add_constraint(self, label, threshold, direction, dose):
		""" TODO: docstring """
		if '<' in direction or '<=' in direction:
			self.structures[label].constraints += D(threshold) <= dose
		else:
			self.structures[label].constraints -= D(threshold) <= dose

		return self.structures[label].constraints.last_key

	def drop_constraint(self, constr_id):
		""" TODO: docstring """
		for s in self.structures.itervalues():
			s.constraints -= constr_id

	def clear_constraints(self):
		""" TODO: docstring """
		for s in self.structures.values():
			s.constraints.clear()

	def change_constraint(self, constr_id, threshold, direction, dose):
		for s in self.structures.values():
			s.set_constraint(constr_id, threshold, direction, dose)

	def change_objective(self, label, dose=None, w_under=None, w_over=None):
		""" TODO: docstring """
		if self.structures[label].is_target:
			self.structures[label].dose = dose
			self.structures[label].w_under = w_under
		self.structures[label].w_over = w_over

	def plan(self, **options):
		""" TODO: docstring """
		if self.physics is None:
			raise AttributeError('case.physics must be intialized '
								 'before case.plan() can be called')
		elif self.anatomy is None:
			raise AttributeError('case.anatomy must be intialized '
								 'before case.plan() can be called')
		elif self.prescription is None:
			raise AttributeError('case.prescription must be intialized '
								 'before case.plan() can be called')

		# use 2 pass OFF by default
		# dvh slack ON by default
		use_2pass = options['dvh_exact'] = options.pop('dvh_exact', False)
		use_slack = options['dvh_slack'] = options.pop('dvh_slack', True)

		# objective weight for slack minimization
		gamma = options['gamma'] = options.pop('slack_penalty', None)

		self.__run = RunRecord(
				self.structures,
				use_2pass=use_2pass,
				use_slack=use_slack,
				gamma=gamma)

		# solve problem
		self.problem.solve(self.structures, self.__run.output, **options)

		# update doses
		if self.feasible:
			self.calc_doses()
			return True
		else:
			warn('Problem infeasible as formulated')
			return False

	def calc_doses(self, x=None):
		""" TODO: docstring """
		x_ = x if x else self.x
		if x_ is None:
			raise ValueError('provide a beam intensity vector or run '
							 'optimization at least once to calculate '
							 'doses')

		for s in self.structures.values():
			s.calc_y(x_)


	def x_num_nonzero(self, tolerance=1e-6):
		return sum(self.x > tolerance) if self.x else 0


	def property_check(requested):
		if self.__run is None:
			raise AttributeError('cannot retrieve property {} without '
								 'performing at least one optimization '
								 'run'.format(requested))

	@property
	def solver_info(self):
		self.property_check('solver_info')
		return self.__run.info

	@property
	def x(self):
		self.property_check('x')
		return self.__run.x_exact if self.__run.x_exact else self.__run.x

	@property
	def x_pass1(self):
		self.property_check('x_pass1')
		return self.__run.x

	@property
	def x_pass2(self):
		self.property_check('x_pass2')
		return self.__run.x_exact

	@property
	def solvetime(self):
		self.property_check('solvetime')
		tm1 = self.solvetime_pass1
		tm2 = self.solvetime_pass2
		return tm1 + tm2 if (tm1 and tm2) else tm1

	@property
	def solvetime_pass1(self):
		self.property_check('solvetime_pass1')
		return self.__run.solvetime

	@property
	def solvetime_pass2(self):
		self.property_check('solvetime_pass2')
		return self.__run.solvetime_exact

	@property
	def feasible(self):
		self.property_check('feasible')
		return self.__run.feasible

	@property
	def plotting_data(self):
		""" TODO: docstring """
 		d = {}
		for label, s in self.structures.iteritems():
			d[label] = s.plotting_data
		return d

	def get_plotting_data(self, **options):
		""" TODO: docstring """
		calc = options.pop('calc', False)
		firstpass = options.pop('firstpass', False)
		x = options.pop('x', None)

 		if calc and isinstance(x, ndarray):
			if x.size == self.n_beams:
				self.calc_doses(squeeze(x))
			else:
				raise ValueError('argument "x" must be a numpy.ndarray'
								 'of length {}, ignoring argument'.format(
								 self.n_beams))
		elif calc and firstpass:
			self.calc_doses(self.x_pass1)
		elif calc:
			self.calc_doses(self.x)
		return self.plotting_data

	@property
	def n_structures(self):
		return self.anatomy.n_structures

	@property
	def n_voxels(self):
		return self.physics.voxels

	@property
	def n_beams(self):
		return self.phsyics.beams