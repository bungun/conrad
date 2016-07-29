from numpy import ndarray, array, squeeze, zeros, ones, nan
from scipy.sparse import csr_matrix, csc_matrix

from conrad.compat import *
from conrad.defs import CONRAD_DEBUG_PRINT, positive_real_valued, \
						sparse_or_dense, vec
from conrad.physics.units import cm3, Gy, DeliveredDose
from conrad.medicine.dose import Constraint, MeanConstraint, ConstraintList, \
								 PercentileConstraint, DVH, RELOPS

"""
TODO: structure.py docstring
"""

W_UNDER_DEFAULT = 1.
W_OVER_DEFAULT = 0.05
W_NONTARG_DEFAULT = 0.025

class Structure(object):
	""" TODO: docstring """
	def __init__(self, label, name, is_target, size=None, **options):
		""" TODO: docstring """
		# basic information
		if not isinstance(label, (int, str)):
			raise TypeError('argument "label" must be of type {} or {}'
							''.format(int, str))
		self.label = label
		self.name = str(name)
		self.is_target = bool(is_target)
		self.__size = None
		self.__dose = 0. * Gy
		self.__boost = 1.
		self.__w_under = nan
		self.__w_over = nan
		self.__A_full = None
		self.__A_mean = None
		self.__voxel_weights = None
		self.__y = None
		self.__y_mean = nan
		self.dvh = None
		self.constraints = ConstraintList()

		if size is not None:
			self.size = size
		if is_target:
			self.dose = options.pop('dose', 1. * Gy)
		self.A_full = options.pop('A', None)
		self.A_mean = options.pop('A_mean', None)

		WU_DEFAULT = W_UNDER_DEFAULT if is_target else 0.
		WO_DEFAULT = W_OVER_DEFAULT if is_target else W_NONTARG_DEFAULT

		self.w_under = options.pop('w_under', WU_DEFAULT)
		self.w_over = options.pop('w_over', WO_DEFAULT)

	@property
	def plannable(self):
		size_set = positive_real_valued(self.size)
		full_mat_usable = sparse_or_dense(self.A_full)
		if full_mat_usable:
			full_mat_usable &= self.size == self.A_full.shape[0]

		collapsed_mat_usable = bool(
				isinstance(self.A_mean, ndarray) and self.collapsable)

		usable_matrix_loaded = full_mat_usable or collapsed_mat_usable
		return size_set and usable_matrix_loaded

	@property
	def size(self):
		return self.__size

	@size.setter
	def size(self, size):
		if not positive_real_valued(size):
			raise ValueError('argument "size" must be a positive int')
		else:
			self.__size = int(size)
			self.dvh = DVH(self.size)

			# default to uniformly weighted voxels
			self.voxel_weights = ones(self.size)

	def reset_matrices(self):
		self.__A_full = None
		self.__A_mean = None

	@property
	def collapsable(self):
		return self.constraints.mean_only and not self.is_target

	@property
	def A_full(self):
		return self.__A_full

	@A_full.setter
	def A_full(self, A_full):
		if A_full is None:
			return

		# verify type of A_full
		if not sparse_or_dense(A_full):
			raise TypeError('input A must by a numpy or scipy csr/csc '
							'sparse matrix')

		if self.size is not None:
			if A_full.shape[0] != self.size:
				raise ValueError('# rows of "A_full" must correspond to value '
								 ' of property size ({}) of {} object'.format(
								 self.size, Structure))
		else:
			self.size = A_full.shape[0]

		self.__A_full = A_full
		self.A_mean = None

	@property
	def A_mean(self):
		return self.__A_mean

	@A_mean.setter
	def A_mean(self, A_mean=None):
		if A_mean is not None:
			if not isinstance(A_mean, ndarray):
				raise TypeError('if argument "A_mean" is provided, it must be '
								'of type {}'.format(ndarray))
			elif not A_mean.size in A_mean.shape:
				raise ValueError('if argument "A_mean" is provided, it must be'
								 ' a row or column vector. shape of argument: '
								 '{}'.format(A_mean.shape))
			else:
				if self.__A_full is not None:
					if len(A_mean) != self.__A_full.shape[1]:
						raise ValueError('field "A_full" already set; '
										 'proposed value for "A_mean" '
										 'must have same number of entries '
										 '({}) as columns in A_full ({})'
										 ''.format(len(A_mean),
										 self.__A_full.shape[1]))
				self.__A_mean = vec(A_mean)
		elif self.__A_full is not None:
			if not isinstance(self.A_full, (ndarray, csc_matrix, csr_matrix)):
				raise TypeError('cannot calculate structure.A_mean from'
								'structure.A_full: A_full must be one of'
								' ({},{},{})'.format(ndarray, csc_matrix,
								csr_matrix))
			else:
				self.__A_mean = self.A_full.sum(0) / self.A_full.shape[0]
				if not isinstance(self.A_full, ndarray):
					self.__A_mean = vec(self.__A_mean)

	@property
	def A(self):
		return self.__A_full

	@property
	def voxel_weights(self):
		return self.__voxel_weights

	@voxel_weights.setter
	def voxel_weights(self, weights):
		if self.size in (None, nan, 0):
			raise ValueError('structure size must be defined to add '
							 'voxel weights')
		if len(weights) != self.size:
			raise ValueError('length of input "weights" ({}) does not '
							 'match structure size ({}) of this {} '
							 'object'
							 ''.format(len(weights), self.size, Structure))
		if any(weights < 0):
			raise ValueError('negative voxel weights not allowed')
		self.__voxel_weights = vec(weights)

	def set_constraint(self, constr_id, threshold=None, relop=None, dose=None):
		if constr_id in self.constraints.items:
			if isinstance(self.constraints[constr_id], PercentileConstraint) \
					and threshold is not None:
				self.constraints[constr_id].percentile = threshold
			if relop is not None:
				self.constraints[constr_id].relop = relop
			if dose is not None:
				self.constraints[constr_id].dose = dose
		else:
			raise ValueError('contraint with ID {} not found in constraints '
							 'attached to this {}'.format(constr_id,
							 Structure))

	@property
	def dose_rx(self):
		return self.__dose

	@property
	def dose(self):
		return self.__boost * self.__dose

	@dose_rx.setter
	def dose_rx(self, dose):
		if not self.is_target: return
		if not isinstance(dose, DeliveredDose):
			raise TypeError('argument "dose" must be of type {}'
							''.format(DeliveredDose))
		self.__dose = dose
		self.__boost = 1.

	@dose.setter
	def dose(self, dose):
		if not self.is_target: return
		if not isinstance(dose, DeliveredDose):
			raise TypeError('argument "dose" must be of type {}'
							''.format(DeliveredDose))
		if dose.value == 0:
			raise ValueError('zero dose invalid for target structure')
		if self.__dose.value == 0:
			self.__dose = dose
			self.__boost = 1.
		else:
			self.__boost = dose.to_Gy.value / self.__dose.to_Gy.value

	@property
	def dose_unit(self):
		u = 1 * self.__dose
		u.value = 1
		return u

	@property
	def w_under(self):
		""" TODO: docstring """
		if not positive_real_valued(self.size):
			return nan

		if isinstance(self.__w_under, (float, int)):
		    return self.__w_under / float(self.size)
		else:
			return None

	@property
	def w_under_raw(self):
	    return self.__w_under

	@w_under.setter
	def w_under(self, weight):
		if not self.is_target:
			self.__w_under = 0.
			return

		if isinstance(weight, (int, float)):
			self.__w_under = max(0., float(weight))
			if weight < 0:
				raise ValueError('negative objective weights not allowed')
		else:
			raise TypeError('argument "weight" must be a float >= 0')

	@property
	def w_over(self):
		""" TODO: docstring """
		if not positive_real_valued(self.size):
			return nan

		if isinstance(self.__w_over, (float, int)):
		    return self.__w_over / float(self.size)
		else:
			return None

	@property
	def w_over_raw(self):
	    return self.__w_over

	@w_over.setter
	def w_over(self, weight):
		if isinstance(weight, (int, float)):
			self.__w_over = max(0., float(weight))
			if weight < 0:
				raise ValueError('negative objective weights not allowed')
		else:
			raise TypeError('argument "weight" must be a float >= 0')

	def calculate_dose(self, beam_intensities):
		self.calc_y(beam_intensities)

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
		if isinstance(self.__y_mean, ndarray):
			self.__y_mean = self.__y_mean[0]

		# make DVH curve from calculated dose
		self.dvh.data = self.__y

	@property
	def y(self):
		""" TODO: docstring """
		return self.__y

	@property
	def mean_dose(self):
		""" TODO: docstring """
		return self.__y_mean * self.dose_unit

	@property
	def min_dose(self):
		""" TODO: docstring """
		if self.dvh is None:
			return nan * Gy
		return self.dvh.min_dose * self.dose_unit

	@property
	def max_dose(self):
		""" TODO: docstring """
		if self.dvh is None:
			return nan * Gy
		return self.dvh.max_dose * self.dose_unit

	def satisfies(self, constraint):
		if self.dvh is None:
			raise ValueError('structure DVH does not exist, cannot evaluate '
							 'constraint satisfaction.\n(assign structure '
							 'size explicitly by setting field "{}.size"\nor '
							 'impicitly by assigning a dose matrix with '
							 'field "{}.A_full"\nto trigger DVH instantiation)'
							 ''.format(Structure, Structure))
		if not self.dvh.populated:
			raise ValueError('structure DVH not populated by dose data, '
							 'cannot evaluate constraint satisfaction\n'
							 '(assign dose by setting field "{}.y")'
							 ''.format(Structure))

		if not isinstance(constraint, Constraint):
			raise TypeError('argument "constraint" must be of type '
				'conrad.dose.Constraint')

		dose = constraint.dose.value
		relop = constraint.relop

		if isinstance(constraint.threshold, str):
			if constraint.threshold == 'mean':
				dose_achieved = self.mean_dose
			elif constraint.threshold == 'min':
				dose_achieved = self.min_dose
			elif constraint.threshold == 'max':
				dose_achieved = self.max_dose
		else:
			dose_achieved = self.dvh.dose_at_percentile(
				constraint.threshold)

		if relop == RELOPS.LEQ:
			status = dose_achieved <= dose
		elif relop == RELOPS.GEQ:
			status = dose_achieved >= dose

		return (status, dose_achieved)

	@property
	def plotting_data(self):
		""" return plotting data from DVH curve and constraints, as well
	 		as the prescribed dose
	 	"""
		return {'curve': self.dvh.plotting_data,
				'constraints': self.constraints.plotting_data,
				'rx': self.dose_rx,
				'target': self.is_target}

	@property
	def __header_string(self):
		""" return header string for structure, comprising name and
			label
		"""
		out = '\nStructure: '
		if self.name != '':
			out += '{}'.format(self.name)
		else:
			out += '<unnamed>'
		out += ' (label = {})\n'.format(self.label)
		return out

	@property
	def __obj_string(self):
		""" return string of objectives attached to Structure object """
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
		""" return string of constraints attached to Structure object """
		out = ''
		for key in self.constraints.items:
			out += self.constraints[key].__str__()
			out += '\n'
		return out

	def summary(self, percentiles=[2, 25, 75, 98]):
		""" given list- or array-like argument percentiles, retrieve and
			return a dictionary of doses at each percentile, as well as
			the MEAN, MIN and MAX doses
		"""
		s = {}
		s['mean'] = self.mean_dose
		s['min'] = self.min_dose
		s['max'] = self.max_dose
		for p in percentiles:
			s['D' + str(p)] = self.dvh.dose_at_percentile(p) * self.dose_unit
		return s

	@property
	def __summary_string(self):
		""" return string of MEAN, MIN, and MAX doses, as well as doses
			at several default percentiles: 98%, 75%, 25%, 2%
		"""
		summary = self.summary()
		hdr = '{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}\n'.format(
				'mean', 'min', 'max', 'D98', 'D75', 'D25', 'D2')
		vals = str('{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}\n'.format(
				summary['mean'], summary['min'], summary['max'],
				summary['D98'], summary['D75'], summary['D25'], summary['D2']))
		return hdr + vals

	@property
	def objective_string(self):
		""" print structure header and objectives """
		return self.__header_string + self.__obj_string

	@property
	def constraints_string(self):
		""" prinst structure header and constraints """
		return self.__header_string + self.__constr_string

	@property
	def summary_string(self):
		""" print structure header and dose summary """
		return self.__header_string + self.__summary_string

	def __str__(self):
		""" print structure header, objectives, and constraints """
		return self.__header_string + self.__obj_string + self.__constr_string