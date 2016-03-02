from numpy import ndarray, array, squeeze, zeros, nan
from scipy.sparse import csr_matrix, csc_matrix
from conrad.dose import Constraint, MeanConstraint, ConstraintList, DVH, DIRECTIONS
from conrad.defs import CONRAD_DEBUG_PRINT

"""
TODO: structure.py docstring
"""

class Structure(object):
	W_UNDER_DEFAULT = 1.
	W_OVER_DEFAULT = 0.05
	W_NONTARG_DEFAULT = 0.1

	""" TODO: docstring """
	def __init__(self, label, name, is_target, dose, **options):
		""" TODO: docstring """
		# basic information
		self.label = label
		self.name = name
		self.is_target = is_target
		self.__size = None
		self.__dose = max(float(dose), 0.) if is_target else 0.
		self.__boost = 1.
		self.__w_under = 0.
		self.__w_over = 0.
		self.__A_full = None
		self.__A_mean = None
		self.__A_clustered = None
		self.__voxels_2_clusters = None
		self.__cluster_sizes = None
		self.__representation = 'full'
		self.__y = None
		self.__y_mean = nan

		# number of voxels in structure
		self.size = options.pop('size', None)

		# dose constraints
		self.constraints = ConstraintList()
		
		# dvh curve
		self.dvh = DVH(self.size) if self.size is not None else None

		# set (pointer to) subsection of dose matrix corresponding to structure
		self.A_full = options.pop('A', None)

		# set (pointer to) clustered version of same dose matrix
		self.A_clustered = (options.pop('A_clustered', None),
			options.pop('vox2cluster', None),
			options.pop('vox_per_cluster', None))

		# set (pointer to) fully compressed version of same dose matrix
		self.A_mean = options.pop('A_mean', None)

		# objective weights (set to defaults if not provided)
		w_under_default = self.W_UNDER_DEFAULT if self.is_target else 0.
		w_over_default = self.W_OVER_DEFAULT if self.is_target else self.W_NONTARG_DEFAULT
		self.w_under = options.pop('w_under', w_under_default)
		self.w_over = options.pop('w_over', w_over_default)

	@property
	def size(self):
		return self.__size

	@size.setter
	def size(self, size):
		if isinstance(size, (int, float)):
			if size <= 0:
				ValueError('argument "size" must be a positive int')
			else:
				self.__size = int(size)
				self.dvh = DVH(self.size)
		else:
			TypeError('argument "size" must be a positive int')

	@property
	def collapsable(self):
		return self.constraints.mean_only and not self.is_target

	@property
	def A_full(self):
		return self.__A_full
	
	@A_full.setter
	def A_full(self, A_full):
		# verify type of A_full
		if A_full is not None and not isinstance(A_full, 
			(ndarray, csr_matrix, csc_matrix)):
			TypeError("input A must by a numpy or "
				"scipy csr/csc sparse matrix")

		self.__A_full = A_full


	@property
	def A_mean(self):
		return self.__A_mean

	@A_mean.setter
	def A_mean(self, A_mean = None):
		if A_mean is not None:
			self.__A_mean = A_mean
		elif self.A_full is not None:
			self.__A_mean = self.A_full.sum(0) / self.A_full.shape[0]
			if not isinstance(self.A_full, ndarray):
				# (handling for sparse matrices)
				self.__A_mean = squeeze(array(self.__A_mean)) 

	@property
	def A_clustered(self):
		return (self.__A_clustered, self.__voxels_2_clusters,
			self.__cluster_sizes)

	@A_clustered.setter
	def A_clustered(self, input_tuple):
		valid = True
		input_len = len(input_tuple)

		valid &= input_len >= 2
		valid &= isinstance(input_tuple[0], (ndarray, csr_matrix, csc_matrix))
		valid &= isinstance(input_tuple[1], (ndarray, list))
		if input_len > 2:
			valid &= isinstance(input_tuple[2], (ndarray, list))

		if not valid:
			ValueError('must provide at least (1) the clustered matrix and '
				'(2) the voxel->cluster mapping to add clustered matrix; '
				'optionally provide (3) the vector/list of cluster sizes')

		A_clu = input_tuple[0]
		vox2cluster = input_tuple[1]
		cluster_sizes = input_tuple[2] if input_len > 2 else None

		self.__A_clustered = A_clu
		self.__voxels_2_clusters = vox2cluster

		if cluster_sizes is not None:
			self.__cluster_sizes = cluster_sizes
		elif vox2cluster is not None:
			self.__cluster_sizes = zeros(A_clu.shape[0])
			for cluster in self.__voxels_2_clusters:
				self.vpc[cluster] += 1

	@property
	def A(self):
		return self.__A_full
	
	def set_objective(self, dose, w_under, w_over):
		if self.is_target:
			self.dose = dose
			self.w_under = w_under
		self.w_over = w_over

	def set_constraint(self, constr_id, threshold, direction, dose):
		if self.has_constraint(constr_id):
			c = self.constraints.items[constr_id]
			if isinstance(c, PercentileConstraint): 
				c.percentile = threshold
			c.direction = direction
			c.dose = dose
			self.constraints.items[constr_id] = c

	@property
	def dose_rx(self):
		return self.__dose

	@property
	def dose(self):
		return self.__dose * self.__boost

	@dose.setter
	def dose(self, dose):
		if not self.is_target: return
		if isinstance(dose, (int, float)):
			self.__boost = max(0., float(dose)) / self.__dose
			if dose < 0:
				ValueError('negative doses are unphysical and '
					'not allowed in dose constraints')
		else:
			TypeError('argument "weight" must be a float '
				'with value >= 0')

	@property
	def w_under(self):
		""" TODO: docstring """
		if isinstance(self.__w_under, (float, int)):
		    return self.__w_under / float(self.size)
		else:
			return None
	
	@property
	def w_under_raw(self):
	    return self.__w_under
	
	@w_under.setter
	def w_under(self, weight):
		if isinstance(weight, (int, float)):
			self.__w_under = max(0., float(weight))
			if weight < 0:
				ValueError('negative objective weights not allowed')
		else:
			TypeError('argument "weight" must be a float '
				'with value >= 0')

	@property
	def w_over(self):
		""" TODO: docstring """
		if isinstance(self.__w_over, (float, int)):
		    return self.__w_over / float(self.size)
		else:
			return None
	
	@property
	def w_over_raw(self):
	    return self.__w_under

	@w_over.setter
	def w_over(self, weight):
		if isinstance(weight, (int, float)):
			self.__w_over = max(0., float(weight))
			if weight < 0:
				ValueError('negative objective weights not allowed')
		else:
			TypeError('argument "weight" must be a float '
				'with value >= 0')

	def calc_y(self, x):
		""" TODO: docstring """

		# calculate dose from input vector x:
		# 	y = Ax
		x = squeeze(array(x))
		if isinstance(self.A, (csr_matrix, csc_matrix)):
			self.__y = squeeze(self.A * x)
		elif isinstance(self.A, ndarray):
			self.__y = self.A.dot(x)

		self.__y_mean = self.A_mean.dot(x)

		# make DVH curve from calculated dose
		self.dvh.data = self.__y

	@property
	def y(self):
		""" TODO: docstring """
		return self.__y
	
	@property
	def mean_dose(self):
		""" TODO: docstring """
		return self.__y_mean

	@property
	def min_dose(self):
		""" TODO: docstring """
		return self.dvh.min_dose

	@property
	def max_dose(self):
		""" TODO: docstring """
		return self.dvh.max_dose

	def satisfies(self, constraint):
		if not isinstance(constraint, Constraint):
			raise TypeError('argument "constraint" must be of type '
				'conrad.dose.Constraint')

		dose = constraint.dose
		direction = constraint.direction

		if constraint.threshold == 'mean':
			dose_achieved = self.mean_dose
		elif constraint.threshold == 'min':
			dose_achieved = self.min_dose
		elif constraint.threshold == 'max':
			dose_achieved = self.max_dose
		else:
			dose_achieved = self.dvh.dose_at_percentile(
				constraint.threshold)

		if direction == DIRECTIONS.LEQ:
			status = dose_achieved <= dose
		elif direction == DIRECTIONS.GEQ:
			status = dose_achieved >= dose

		return (status, dose_achieved)

	@property
	def plotting_data(self):
		""" TODO: docstring """
		return {'curve': self.dvh.plotting_data, 
		'constraints': self.constraints.plotting_data}
	
	@property
	def __header_string(self):
		""" TODO: docstring """
		out = 'Structure: {}'.format(self.label)
		if self.name != '':
			out += ' ({})'.format(self.name)
			out += '\n'
		return out		

	@property
	def __obj_string(self):
		""" TODO: docstring """
		out = 'target? {}\n'.format(self.is_target)
		out += 'rx dose: {}\n'.format(self.dose)
		if self.is_target:
			out += 'weight_under: {}\n'.format(self.__w_under)
			out += 'weight_over: {}\n'.format(self.__w_over)			
		else:
			out += 'weight: {}\n'.format(self.__w_over)
		out += "\n"		
		return out

	@property
	def __constr_string(self):
		""" TODO: docstring """
		out = ''
		for dc in self.constraints.itervalues():
			out += dc.__str__()
		out += '\n'
		return out

	def summary(self, percentiles = [2, 25, 75, 98]):
		s = {}
		s['mean'] = self.mean_dose
		s['min'] = self.min_dose
		s['max'] = self.max_dose
		for p in percentiles:
			s['D' + str(p)] = self.dvh.dose_at_percentile(p)
		return s

	@property
	def __summary_string(self):
		summary = self.summary()
		hdr = 'mean | min  | max  | D98  | D75  | D25  | D2   \n'
		vals = str('{:0.2f} | {:0.2f} | {:0.2f} '
			'| {:0.2f} | {:0.2f} | {:0.2f} | {:0.2f}\n'.format(
			summary['mean'], summary['min'], summary['max'],
			summary['D98'], summary['D75'], summary['D25'], summary['D2']))
		return hdr + vals

	@property
	def objective_string(self):
		""" TODO: docstring """
		return self.__header_string + self.__obj_string

	@property
	def constraints_string(self):
		""" TODO: docstring """
		return self.__header_string + self.__constr_string

	@property
	def summary_string(self):
		""" TODO: docstring """
		return self.__header_string + self.__summary_string


	def __str__(self):
		return self.__header_string + self.__obj_string + self.__constr_string
