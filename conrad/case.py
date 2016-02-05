from conrad.dvh import DoseConstraint
from conrad.structure import Structure
from conrad.prescription import Prescription
from conrad.problem import PlanningProblem
from conrad.run_data import RunRecord
from conrad.plot import DVHPlot
from operator import add 
from numpy import cumsum

# TODO: unit test
"""
TODO: case.py docstring
"""

def gen_constraint_id(label, constraint_count):
	""" TODO: docstring """
	return "sid:{}:cid:{}".format(label, constraint_count)

def constraint2label(constr_id):
	""" TODO: docstring """
	return constr_id.split(':')[1]

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
		ValueError("length of vector voxel_labels and "
			"number of rows in dose_matrix must be equal.")

	structures = prescription.structure_dict
	ptr1 = ptr2 = 0

	for l in label_order:
		# obtain structure size
		size = reduce(add, map(lambda v : v == label, voxel_labels))
		structures[l].size = size
		ptr2 += size

		# assess sorting of label blocks:
		if not all(map(lambda v: v == label, voxel_labels[ptr1:ptr2])):
			ValueError("inputs voxel_labels and dose_matrix are expected "
				"to be (block) sorted in the order specified by argument "
				"`label_order'. voxel_labels not block sorted.")
		
		# partition dose matrix	into blocks
		structures[l].A_full = structures[l].A = dose_matrix[ptr1:ptr2, :]
		structures[l].set_block_indices(ptr1, ptr2)
		ptr1 = ptr2

	return structures



def transfer_prescription(prescription, structures):
	"""TODO: docstring"""
	pass


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

	def __init__(self, A, voxel_labels, label_order, prescription_raw):
		"""TODO: Case.__init__() docstring"""
		self.problem = PlanningProblem()
		self.voxel_labels = voxel_labels
		
		# digest clinical specification

		self.prescription = Prescription(prescription_raw)

		# parse full mat + data into Structure objects
		self.structures = build_structures(self.prescription, 
			voxel_labels, label_order, A)

		# set of IDs active constraints 
		# (maintained as DVH constraints added/removed)
		self.active_constraint_IDs = set()

		# counts go with life of Case object,
		# even if, e.g., all constraints removed from a run
		self.constraint_count = 0
		self.run_count = 0

		# dose matrix
		self.A = A
		# self.shape = (self.voxels, self.beams) = A.shape


		self.run_records = {}

		# plot setup
		panels_by_structure = {}
		names_by_structure = {}
		for idx, label in enumerate(label_order):
			panels_by_structure[label] = idx
			names_by_structure[label] = self.structures[label].name
		self.plot = DVHPlot(panels_by_structure, names_by_structure)

	def add_dvh_constraint(self, label, dose, fraction, direction):
		""" TODO: docstring """
		self.constraint_count += 1
		constr = DoseConstraint(dose, fraction, direction)
		cid = gen_constraint_id(label, self.constraint_count)
		self.structures[label].add_constraint(cid, constr)
		self.active_constraint_IDs.add(cid)
		return cid

	def drop_dvh_constraint(self, constr_id):
		""" TODO: docstring """
		label = constraint2label(constr_id)
		self.structures[label].remove_constraint(constr_id)
		self.active_constraint_IDs.remove(constr_id)

	def drop_all_dvh_constraints(self):
		""" TODO: docstring """
		for cid in self.active_constraint_IDs:
			self.drop_dvh_constraint(cid)

	def add_all_rx_constraints(self):
		""" TODO: docstring """
		contraint_dict = self.prescription.constraints_by_label
		for label, constr_list in constraint_dict.iteritems():
			for constr in constr_list:
				self.add_dvh_constraint(label, *constr)

	def drop_all_but_rx_constraints(self):
		""" TODO: docstring """
		self.drop_all_dvh_constraints()
		self.add_all_rx_constraints()

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

		# save output
		self.run_count += 1
		self.run_records[run_count] = rr

		if 'plot' in args:
			self.plot.plot(self.plotting_data)
			if 'plotfile' in kwargs:
				self.plot.save(kwargs['plotfile'])


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
	def n_voxels(self):
		""" TODO: docstring """
		return self.A.shape[0]

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