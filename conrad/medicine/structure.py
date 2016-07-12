from numpy import ndarray, array, squeeze, zeros, nan
from scipy.sparse import csr_matrix, csc_matrix

from conrad.compat import *
from conrad.dose import Constraint, MeanConstraint, ConstraintList, DVH, RELOPS
from conrad.defs import CONRAD_DEBUG_PRINT

"""
TODO: structure.py docstring
"""

W_UNDER_DEFAULT = 1.
W_OVER_DEFAULT = 0.05
W_NONTARG_DEFAULT = 0.025

class Structure(object):
	""" TODO: docstring """
	def __init__(self, label, name, is_target, **options):
		""" TODO: docstring """
		# basic information
		self.label = label
		self.name = name
		self.is_target = bool(is_target)
		self.__size = nan
		self.__dose = 0.
		self.__boost = 1.
		self.__w_under = nan
		self.__w_over = nan
		self.__A_full = None
		self.__A_mean = None
		self.__y = None
		self.__y_mean = nan

		if 'size' in options:
			self.size = options['size']
		self.dose = options.pop('dose', 0.)
		self.A_full = options.pop('A', None)
		self.A_mean = options.pop('A_mean', None)

		WU_DEFAULT = W_UNDER_DEFAULT if is_target else 0.
		WO_DEFAULT = W_OVER_DEFAULT if is_target else W_NONTARG_DEFAULT

		self.w_under = options.pop('w_under', WU_DEFAULT)
		self.w_over = options.pop('w_over', WO_DEFAULT)

		self.constraints = ConstraintList()
		self.dvh = DVH(self.size) if self.size is not None else None

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
			if not isinstance(A_mean, ndarray):
				raise TypeError('if argument "A_mean" is provided, it must be '
								'of type {}'.format(ndarray))
			elif not A_mean.size in A_mean.shape:
				raise ValueError('if argument "A_mean" is provided, it must be'
								 ' a row or column vector. shape of argument: '
								 '{}'.format(A_mean.shape))
			else:
				self.__A_mean = squeee(array(A_mean))
		elif self.A_full is not None:
			if not isinstance(self.A_full, (ndarray, csc_matrix, csr_matrix)):
				raise TypeError('cannot calculate structure.A_mean from'
								'structure.A_full: A_full must be one of'
								' ({},{},{})'.format(ndarray, csc_matrix,
								csr_matrix))
			else:
				self.__A_mean = self.A_full.sum(0) / self.A_full.shape[0]
				if not isinstance(self.A_full, ndarray):
					self.__A_mean = squeeze(array(self.__A_mean))

	@property
	def A(self):
		return self.__A_full

	def set_constraint(self, constr_id, threshold, relop, dose):
		if self.has_constraint(constr_id):
			c = self.constraints.items[constr_id]
			if isinstance(c, PercentileConstraint):
				c.percentile = threshold
			c.relop = relop
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
				raise ValueError('negative objective weights not allowed')
		else:
			raise TypeError('argument "weight" must be a float >= 0')

	@property
	def w_over(self):
		""" TODO: docstring """
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
		relop = constraint.relop

		if constraint.threshold == 'mean':
			dose_achieved = self.mean_dose
		elif constraint.threshold == 'min':
			dose_achieved = self.min_dose
		elif constraint.threshold == 'max':
			dose_achieved = self.max_dose
		else:
			dose_achieved = self.dvh.dose_at_percentile(
				constraint.threshold)

		if relop == DIRECTIONS.LEQ:
			status = dose_achieved <= dose
		elif relop == DIRECTIONS.GEQ:
			status = dose_achieved >= dose

		return (status, dose_achieved)

	@property
	def plotting_data(self):
		""" TODO: docstring """
		return {'curve': self.dvh.plotting_data,
				'constraints': self.constraints.plotting_data,
				'rx': self.dose_rx}

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

	def summary(self, percentiles=[2, 25, 75, 98]):
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