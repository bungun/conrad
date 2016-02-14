from conrad.prescription import Prescription
from conrad.problem import PlanningProblem
from conrad.run_data import RunRecord, PlanningHistory
from operator import add 
from numpy import ndarray, array, squeeze, zeros
from warning import warn

"""
TODO: case.py docstring
"""

def default_weights(is_target = False):
	""" TODO: docstring """
	if is_target:
		# w_under = 1, w_over = 0.05
		return 1., 0.05
	else:
		return None, 0.1

def build_structures(prescription, voxel_labels, label_order, dose_matrix):
	"""TODO: docstring

	NB: ASSUMES dose_matrix IS SORTED IN SAME ORDER AS LABELS IN voxel_labels
	
	(fails if voxel_labels unsorted; TODO: sorting? pre-sort?)
	"""

	if not dose_matrix.shape[0] == len(voxel_labels):
		raise ValueError("length of vector voxel_labels and "
			"number of rows in dose_matrix must be equal.")

	structures = prescription.structure_dict
	ptr1 = ptr2 = 0

	for label in label_order:
		# obtain structure size
		size = reduce(add, map(lambda v : int(v == label), voxel_labels))
		structures[label].size = size
		ptr2 += size
		
		# assess sorting of label blocks:
		if not all(map(lambda v: v == label, voxel_labels[ptr1:ptr2])):
			raise ValueError("inputs voxel_labels and dose_matrix are expected "
							 "to be (block) sorted in the order specified by argument "
							 "`label_order'. voxel_labels not block sorted.")
		
		# partition dose matrix	into blocks
		structures[label].A_full = dose_matrix[ptr1:ptr2, :]
		structures[label].A_mean = squeeze(array(
			csum(dose_matrix[ptr1:ptr2, :], 0))) / size
		ptr1 = ptr2

	return structures


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

	def __init__(self, A, voxel_labels, label_order, 
		prescription_raw, suppress_rx_constraints = False):
		"""TODO: Case.__init__() docstring"""
		self.problem = PlanningProblem()
		self.voxel_labels = voxel_labels
		self.label_order = label_order

		# digest clinical specification
		self.prescription = Prescription(prescription_raw)

		# parse full mat + data into Structure objects
		self.structures = build_structures(self.prescription, 
			voxel_labels, label_order, A)

		# append prescription constraints unless suppressed:
		if not suppress_rx_constraints:
			self.add_all_rx_constraints()

		# dose matrix
		self.A = A

		# planning history
		self.history = PlanningHistory()



	def add_dvh_constraint(self, label, threshold, direction, dose):
		""" TODO: docstring """
		if '<' in direction:
			self.structures[label].constraints += D(threshold) <= dose
		else:
			self.structures[label].constraints -= D(threshold) <= dose

		return self.structures[label].constraints.last_key

	def drop_dvh_constraint(self, constr_id):
		""" TODO: docstring """
		for s in self.structures.itervalues():
			s.constraints -= constr_id

	def drop_all_dvh_constraints(self):
		""" TODO: docstring """
		for s in self.structures.itervalues():
			s.constraints.clear()

	def add_all_rx_constraints(self):
		""" TODO: docstring """
		constraint_dict = self.prescription.constraints_by_label
		for label, constr_list in constraint_dict.iteritems():
			for constr in constr_list:
				self.structures[label].constraints += constr

	def drop_all_but_rx_constraints(self):
		""" TODO: docstring """
		self.drop_all_dvh_constraints()
		self.add_all_rx_constraints()

	def change_constraint(self, constr_id, threshold, direction, dose):
		for s in self.structures.itervalues():
			s.set_constraint(constr_id, threshold, direction, dose)

	def change_objective(self, label, dose = None, 
		w_under = None, w_over = None):
		""" TODO: docstring """
		self.structures[label].set_objective(dose, w_under, w_over)

	def plan(self, **options):
		""" TODO: docstring """
		# use 2 pass OFF by default
		pass = options['dvh_2pass'] = options.pop('dvh_2pass', False)

		# dvh slack ON by default
		use_slack = ['dvh_no_slack'] if 'dvh_no_slack' in kwargs else False

		# objective weight for slack minimization
		gamma = kwargs['dvh_wt_slack'] if 'dvh_wt_slack' in kwargs else None
		if gamma is not None: kwargs['gamma'] = gamma

		rr = RunRecord(self.structures, 
			use_2pass = use_2pass, 
			use_slack = use_slack, 
			gamma = gamma)


		# solve problem
		self.problem.solve(self.structures, rr.output, **options)

		# save output
		self.history += rr

		# update doses
		if self.feasible:
			self.calc_doses()
			return True
		else:
			warn(Warning('Problem infeasible as formulated'))
			return False

	def calc_doses(self, x = None):
		""" TODO: docstring """
		if x is None: x = self.x
		if x is None:
			ValueError('optimization must be run at least once '
				'to calculate doses')

		for s in self.structures.itervalues():
			s.calc_y(x)

	@property
	def solver_info(self):
		return self.history.last_info

	@property
	def x(self):
		return self.history.last_x_exact if self.history.last_x_exact is not None else self.history.last_x

	@property
	def x_pass1(self):
		return self.history.last_x
	
	@property
	def x_pass2(self):
		return self.x


	@property
	def feasible(self):
		if self.run_count == 0:
			return None
		else:
			return self.run_records[self.run_count].output.feasible

	def dose_summary_data(self, percentiles = [2, 98], stdev = False):
		d = {}
		for label, s in self.structures.iteritems():
			d[label] = s.get_dose_summary(percentiles = percentiles, stdev = stdev)
		return d

	@property
	def dose_summary_string(self):
		out = ''
		for s in self.structures.itervalues():
			out += s.summary_string
		return out

	@property
	def prescription_report(self):
		""" TODO: docstring """
		return self.prescription.report(self.structures)

	@property
	def prescription_report_string(self):
		return self.prescription.report_string(self.structures)

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
				warn(Warning('argument "x" must be a numpy.ndarray'
					'of length {}, ignoring argument'.format(self.n_beams)))
		elif calc and firstpass:
			self.calc_doses(self.x_pass1)
		elif calc:
			self.calc_doses(self.x)
		return self.plotting_data
		
	@property
	def n_structures(self):
		""" TODO: docstring """
		return len(self.structures.keys())
	
	@property
	def n_voxels(self):
		""" TODO: docstring """
		return self.A.shape[0]

	@property
	def n_beams(self):
		""" TODO: docstring """
		return self.A.shape[1]