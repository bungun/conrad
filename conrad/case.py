from conrad.dvh import DoseConstraint, tuple_to_canonical_string
from conrad.structure import Structure
from conrad.prescription import Prescription
from conrad.problem import PlanningProblem
from conrad.run_data import RunRecord
from conrad.plot import DensityPlot, DVHPlot
from operator import add 
from numpy import cumsum, ndarray, squeeze
# from tabulate import tabulate
from collections import OrderedDict

"""
TODO: case.py docstring
"""

def gen_constraint_id(label, constraint_count):
	""" TODO: docstring """
	return "sid:{}:cid:{}".format(label, constraint_count)

def constraint2label(constr_id):
	""" TODO: docstring """
	return int(constr_id.split(':')[1])

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
		structures[label].set_A_full_and_mean(dose_matrix[ptr1:ptr2, :])
		structures[label].set_block_indices(ptr1, ptr2)
		ptr1 = ptr2

	return structures



def build_prescription_report(prescription, structures):
	"""TODO: docstring"""
	rx_constraints = prescription.constraints_by_label
	report = {}
	for label, s in structures.iteritems():
		sat = []
		for constr in rx_constraints[label]:
			status, dose_achieved = s.check_constraint(constr)
			sat.append({'constraint': constr, 
				'status': status, 'dose_achieved': dose_achieved})
		report[label] = sat
	return report

def stringify_prescription_report(report, structures):
	out = ''
	for k, replist in report.iteritems():
		out += 'Structure {}'.format(k)
		if structures[k].name is not None:
			out += '({})'.format(structures[k].name)
		out += ':\n'
		for item in replist:
			out += tuple_to_canonical_string(item['constraint'])
			out += '\tachieved? ' + string(item['status'])
			out += '\tdose at level: ' + string(item['status']) + '\n'

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
		self.run_tags = {}

		# append prescription constraints unless suppressed:
		if not suppress_rx_constraints:
			self.add_all_rx_constraints()

		# dose matrix
		self.A = A
		# self.shape = (self.voxels, self.beams) = A.shape

		self.__DOSES_CALCULATED__ = False
		self.run_records = {}

		# plot setup
		panels_by_structure = {}
		names_by_structure = {}
		for idx, label in enumerate(label_order):
			panels_by_structure[label] = idx+1
			names_by_structure[label] = self.structures[label].name
		self.dvh_plot = DVHPlot(panels_by_structure, names_by_structure)
		self.density_plot = DensityPlot(panels_by_structure, names_by_structure)

	# def add_dvh_constraint(self, label, dose, fraction, direction):
	#	""" TODO: docstring """
	#	self.constraint_count += 1
	#	constr = DoseConstraint(dose, fraction, direction)
	#	cid = gen_constraint_id(label, self.constraint_count)
	#	self.structures[label].add_constraint(cid, constr)
	#	self.active_constraint_IDs.add(cid)
	#	return cid
	
	def add_dvh_constraint(self, label, constr):
		""" TODO: docstring """
		self.constraint_count += 1
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
		constraint_dict = self.prescription.constraints_by_label
		for label, constr_list in constraint_dict.iteritems():
			for constr in constr_list:
				self.add_dvh_constraint(label, *constr)

	def drop_all_but_rx_constraints(self):
		""" TODO: docstring """
		self.drop_all_dvh_constraints()
		self.add_all_rx_constraints()

	def change_dvh_constraint(self, constr_id, dose = None, fraction = None, direction = None):
		label = constraint2label(constr_id)
		self.structures[label].set_constraint(constr_id, dose, fraction, direction)

	def change_objective(self, label, dose = None, 
		w_under = None, w_over = None):
		""" TODO: docstring """
		self.structures[label].set_objective(dose, w_under, w_over)

	def plan(self, use_slack = True, use_2pass = False, *args, **kwargs):
		""" TODO: docstring """

		# check for targets
		if not self.has_targets:
			print str("Warning: plan has no targets."
				"Not running optimization.\n\n")
			return

		# objective weight for slack minimization
		gamma = kwargs['dvh_wt_slack'] if 'dvh_wt_slack' in kwargs else None
		if gamma is not None: kwargs['gamma'] = gamma

		rr = RunRecord(self.structures, 
			use_2pass = use_2pass, 
			use_slack = use_slack, 
			gamma = gamma)

		# solve problem
		self.problem.solve(self.structures, rr.output, use_slack, use_2pass, *args, **kwargs)

		# save output
		self.run_count += 1
		self.run_records[self.run_count] = rr

		# update doses
		if self.feasible:
			x_key = 'x_exact' if use_2pass else 'x'
			self.calc_doses(self.solution_data[x_key])

			draw_plot = kwargs['plot'] if 'plot' in kwargs else False
			show_plot = kwargs['show'] if 'show' in kwargs else draw_plot	# Used to suppress plot during unit testing
			plotfile = kwargs['plotfile'] if 'plotfile' in kwargs else None
			if draw_plot:
				self.plot(show_plot, plotfile)
		else:
			print "Problem infeasible as formulated"
	
	def plot(self, show = True, plotfile = None, plot_2pass = False, **options):
		""" TODO: docstring """
		# plot 1st pass DVH curves on same figure as 2nd pass for comparison
		if plot_2pass and self.run_records[self.run_count].profile.use_2pass:
			plotting_data_first = self.get_plotting_data(calc = True, firstpass = True)
			self.dvh_plot.plot(plotting_data_first, show = False, linestyle = '--', **options)
		
		self.dvh_plot.plot(self.plotting_data, show = show, **options)
		if plotfile is not None:
			print "SAVING"
			self.dvh_plot.save(plotfile)
			print "COMPLETE"
			
	def plot_density(self, show = True, plotfile = None, **options):
		""" TODO: docstring """
		self.density_plot.plot(self.plotting_data_density, show = show, **options)
		if plotfile is not None:
			print "SAVING"
			self.density_plot.save(plotfile)
			print "COMPLETE"

	def calc_doses(self, x):
		""" TODO: docstring """
		self.__DOSES_CALCULATED__ = True

		for s in self.structures.itervalues():
			s.calc_y(x)
	
	@property
	def solver_info(self):
		if self.run_count == 0:
			return None
		return self.run_records[self.run_count].output.solver_info

	@property
	def solution_data(self):
		if self.run_count == 0:
			return None
		else:
			return self.run_records[self.run_count].output.optimal_variables

	@property
	def feasible(self):
		if self.run_count == 0:
			return None
		else:
			return self.run_records[self.run_count].output.feasible

	def __lookup_runrecord(runID):
		if self.run_count == 0:
			return None
		if runID in self.run_tags:
			return self.run_records[self.run_tags[runID]]
		elif runID > 0 and runID <= self.run_count:
			return self.run_records[runID]				

	def get_run(runID = None):
		if runID is None: runID = self.run_count
		return __lookup_runrecord(runID)

	def get_run_profile(runID = None):
		if runID is None: runID = self.run_count
		rr = __lookup_runrecord(runID)
		if rr is not None:
			return rr.profile

	def get_run_output(runID = None):
		if runID is None: runID = self.run_count
		rr = __lookup_runrecord(runID)
		if rr is not None:
			return rr.output

	def summary(self):
		""" TODO: docstring """
		table = OrderedDict({'name': []})
		for s in self.structures.itervalues():
			table['name'] += [s.name]
			for key, val in s.dose_summary.table_data.iteritems():
				if key not in table:
					table[key] = []
				table[key] += [val]
		print table
		# print tabulate(table, headers = "keys", tablefmt = "pipe")

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
		if self.run_count == 0 or not self.__DOSES_CALCULATED__:
			return None

		return build_prescription_report(self.prescription, self.structures)

	@property
	def prescription_report_string(self):
		return stringify_prescription_report(self.prescription_report, self.structures)

	@property
	def objective_data(self):
		o = {}
		for label, s in self.structures.iteritems():
			o[label] = s.objective_data
		return o

	def get_objective_by_label(self, label):
		if label not in self.structures: return None
		return self.structures[label].objective_data


	@property
	def plotting_data(self):
		""" TODO: docstring """
		if self.run_count == 0:
			return None
 
 		d = {}
		for label, s in self.structures.iteritems():
			d[label] = s.plotting_data
		return d

	@property
	def plotting_data_json_serializable(self):
		""" TODO: docstring """
		if self.run_count == 0:
			return None
 
 		d = {}
		for label, s in self.structures.iteritems():
			d[label] = s.plotting_data_json_serializable
		return d
	
	@property
	def plotting_data_density(self):
		""" TODO: docstring """
		if self.run_count == 0:
			return None
		
		d = {}
		for label, s in self.structures.iteritems():
			d[label] = s.plotting_data_density
		return d
		
	@property
	def plotting_data_density_json_serializable(self):
		""" TODO: docstring """
		if self.run_count == 0:
			return None
			
		d = {}
		for label, s in self.structures.iteritems():
			d[label] = s.plotting_data_density_json_serializable
		return d

	@property
	def plotting_data_constraints_only(self):
		""" TODO: docstring """
 		d = {}
		for label, s in self.structures.iteritems():
			d[label] = s.plotting_data_constraints_only
		return d	

	def get_plotting_data(self, serializable = False, 
		calc = False, firstpass = False, x = None):
		""" TODO: docstring """
		if self.run_count == 0:
			return None
 
		if calc and isinstance(x, ndarray):
			if x.size == self.n_beams:
				self.calc_doses(squeeze(x))
		elif calc and not firstpass and 'x_exact' in self.solution_data:
			self.calc_doses(self.solution_data['x_exact'])
		elif calc:
			self.calc_doses(self.solution_data['x'])

		if serializable:
			return self.plotting_data_json_serializable
		else:
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

	@property
	def n_dvh_constraints(self):
		""" TODO: docstring """
		return sum([dc.count for dc in self.dvh_constrs_by_struct])
	
	@property
	def has_targets(self):
		""" TODO: docstring """
		for s in self.structures.itervalues():
			if s.is_target: return True
		return False
