"""
structure.py docstring
"""

class Structure(object):
	def __init__(self, label, **options):
		# basic information
		self.label = label
		self.name = ""
		self.is_target = False

		# number of voxels in structure
		self.size = None

		# prescribed dose
		self.dose = None

		# dictionary of DoseConstraint objects attached to structure
		# TODO: is this keyed by constraint uid? who generates the id, 
		# 	how does it get attached?
		self.dose_constraints = {}

		# dvh curve and constraints data for plotting
		self.dvh_curve = None

		# subsection of dose matrix corresponding to structure
		self.A = None

		# clustered version of same dose matrix, voxels
		# voxel->cluster mapping vector
		# voxel counts per cluster
		self.A_clu = None
		self.v2c = None
		self.voxel_per_cluster = None

		# fully compressed version of same dose matrix
		self.A_lin = None

		# dose vector
		self.y = None

		# objective weights
		self.w_under = None
		self.w_over = None

	# possible methods:
	# calc_dose
	# plot, plot_data, ...
	# some way to emit data for objective, constraints
	def set_slacks(self, slack_dict):
		for key in slack_dict:
			if key in self.dose_constraints
				self.dose_constraints[key].set_actual_dose(slack_dict[key])

