"""
Define :class:`Structure`, building block of :class:`~conrad.medicine.Anatomy`.

Attributes:
	W_UNDER_DEFAULT (float): Default objective weight for underdose
		penalty on target structures.
	W_OVER_DEFAULT (float): Default objective weight for underdose
		penalty on non-target structures.
	W_NONTARG_DEFAULT (float): Default objective weight for overdose
		penalty on non-target structures.
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

from numpy import ndarray, array, squeeze, zeros, ones, nan
from scipy.sparse import csr_matrix, csc_matrix

from conrad.defs import CONRAD_DEBUG_PRINT, positive_real_valued, \
						sparse_or_dense, vec
from conrad.physics.units import cm3, Gy, DeliveredDose
from conrad.medicine.dose import Constraint, MeanConstraint, ConstraintList, \
								 PercentileConstraint, DVH, RELOPS

W_UNDER_DEFAULT = 1.
W_OVER_DEFAULT = 0.05
W_NONTARG_DEFAULT = 0.025

class Structure(object):
	"""
	:class:`Structure` manages the dose information (including the dose
	influence matrix, dose calculations and dose volume histogram), as
	well as optimization objective information---including dose
	constraints---for a set of voxels (volume elements) in the patient
	volume to be treated as a logically homogeneous unit with respect to
	the optimization process.

	There are usually three types of structures:
		- Anatomical structures, such as a kidney or the spinal
			cord, termed organs-at-risk (OARs),
		- Clinically delineated structures, such as a tumor or a target
			volume, and,
		- Tissues grouped together by virtue of not being explicitly
			delineated by a clinician, typically lumped together under
			the catch-all category "body".

	Healthy tissue structures, including OARs and "body", are treated as
	non-target, are prescribed zero dose, and only subject to an
	overdose penalty during optimization.

	Target tissue structures are prescribed a non-zero dose, and subject
	to both an underdose and an overdose penalty.

	Attributes:
		label: (:obj:`int` or :obj:`str`): Label, applied to each voxel
			in the structure, usually generated during CT contouring
			step in the clinical workflow for treatment planning.
		name (:obj:`str`): Clinical or anatomical name.
		is_target (:obj:`bool`): ``True`` if structure is a target.
		dvh (:class:`DVH`): Dose volume histogram.
		constraints (:class:`ConstraintList`): Mutable collection of
			dose constraints to be applied to structure during
			optimization.

		 """
	def __init__(self, label, name, is_target, size=None, **options):
		"""
		Initialize target/non-target :class:`Structure` with label and name.

		Arguments:
			label (:obj:`int` or :obj:`str`): Structure label.
			name: Name of structure (e.g., 'PTV', 'body', or
				'spinal cord').
			is_target (:obj:`bool`): ``True`` if structure is intended
				to receive a non-zero dose level during treatment.
			size (:obj:`int`, optional): Number of voxels (volume
				elements) in structure.
			**options: Arbitrary keyword arguments.

		Raises:
			TypeError: If ``label`` is not an :obj:`int` or :obj:`str`.
		"""
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
		"""
		True if structure's attached data is sufficient for optimization.

		Minimum requirements:
			- Structure size determined, and
			- Dose matrix assigned, *or*
			- Structure collapsable and mean dose matrix assigned.
		"""
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
		"""
		Structure size (i.e., number of voxels in structure).

		Raises:
			ValueError: If ``size`` not an :obj:`int`.
		"""
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
		""" Reset structure's dose and mean dose matrices to ``None`` """
		self.__A_full = None
		self.__A_mean = None

	@property
	def collapsable(self):
		""" ``True`` if optimization can be performed with mean dose only. """
		return self.constraints.mean_only and not self.is_target

	@property
	def A_full(self):
		"""
		Full dose matrix (dimensions = voxels x beams).

		Setter method will perform two additional tasks:
			- If :attr:`Structure.size` is not set, set it based on
				number of rows in ``A_full``.
			- Trigger :attr:`Structure.A_mean` to be calculated from
				:attr:`Structure.A_full`.

		Raises:
			TypeError: If ``A_full`` is not a matrix in
				:class:`ndarray`, :class:`csc_matrix`, or
				:class:`csr_matrix` formats.
			ValueError: If :attr:`Structure.size` is set, and the number
				of rows in ``A_full`` does not match
				:attr:`Structure.size`.
		"""
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
				raise ValueError('# rows of "A_full" must correspond to '
								 'value  of property size ({}) of {} '
								 'object'.format(self.size, Structure))
		else:
			self.size = A_full.shape[0]

		self.__A_full = A_full

		# Pass "None" to self.A_mean setter to trigger calculation of
		# mean dose matrix from full dose matrix.
		self.A_mean = None

	@property
	def A_mean(self):
		"""
		Mean dose matrix (dimensions = ``1`` x ``beams``).

		Setter expects a one dimensional :class:`ndarray` representing
		the mean dose matrix for the structure. If this optional
		argument is not provided, the method will attempt to calculate
		the mean dose from :attr:`Structure.A_full`.

		Raises:
			TypeError: If ``A_mean`` provided and not of type
				:class:`ndarray`, *or* if mean dose matrix is to be
				calculated from :attr:`Structure.A_full`, but full dose
				matrix is not a :mod:`conrad`-recognized matrix type.
			ValueError: If ``A_mean`` is not dimensioned as a row or
				column vector, or number of beams implied by ``A_mean``
				conflicts with number of beams implied by
				:attr:`Structure.A_full`.
		 """
		return self.__A_mean

	@A_mean.setter
	def A_mean(self, A_mean=None):
		if A_mean is not None:
			if not isinstance(A_mean, ndarray):
				raise TypeError('if argument "A_mean" is provided, it '
								'must be of type {}'.format(ndarray))
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
			if not sparse_or_dense(self.A_full):
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
		""" Alias for :attr:`Structure.A_full`. """
		return self.__A_full

	@property
	def voxel_weights(self):
		"""
		Voxel weights, or relative volumes of voxels.

		The voxel weights are the ``1`` vector if the structure volume
		is regularly discretized, and some other set of integer values
		if voxels are clustered.

		Raises:
			ValueError: If :attr:`Structure.voxel_weights` setter called
				before :attr:`Structure.size` is defined, or if length
				of input does not match :attr:`Structure.size`, or if
				any of the provided weights are negative.
		"""
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
		"""
		Modify threshold, relop, and dose of an existing constraint.

		Arguments:
			constr_id (:obj:`str`): Key to a constraint in
				:attr:`Structure.constraints`.
			threshold (optional): Percentile threshold to assign if
				queried constraint is of type
				:class:`PercentileConstraint`, no-op otherwise. Must be
				compatible with the setter method for
				:attr:`PercentileConstraint.percentile`.
			relop (optional): Inequality constraint sense. Must be
				compatible with the setter method for
				:attr:`Constraint.relop`.
			dose (optional): Constraint dose. Must be compatible with
				setter method for :attr:`Constraint.dose`.

		Returns:
			None

		Raises:
			ValueError: If ``constr_id`` is not the key to a constraint
				in the :class:`Constraintlist` located at
				:attr:`Structure.constraints`.
		"""
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
	def dose(self):
		"""
		Dose level targeted in structure's optimization objective.

		The dose has two components: the precribed dose,
		:attr:`Structure.dose_rx`, and a multiplicative adjustment
		factor, :attr:`Structure.boost`.

		Once the structure's dose has been initialized, setting
		:attr:`Structure.dose` will change the adjustment factor. This
		is to distinguish (and allow for differences) between the dose
		level prescribed to a structure by a clinician and the dose
		level request to a numerical optimization algorithm that yields
		a desirable distribution, since the latter may require some
		offset relative to the former. To change the reference dose
		level, use the :attr:`Structure.dose_rx` setter.

		Setter is no-op for non-target structures, since zero dose is
		prescribed always.

		Raises:
			TypeError: If requested dose does not have units of
				:class:`DeliveredDose`.
			ValueError: If zero dose is requested to a target structure.
		"""
		return self.boost * self.__dose

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
	def boost(self):
		"""
		Adjustment factor from precription dose to optimization dose.
		"""
		return self.__boost

	@property
	def dose_rx(self):
		"""
		Prescribed dose level.

		Setting this field sets :attr:`Structure.dose` to the requested
		value and :attr:`Structure.boost` to ``1``.
		"""
		return self.__dose

	@dose_rx.setter
	def dose_rx(self, dose):
		self.__dose.value = 0
		self.dose = dose

	@property
	def dose_unit(self):
		"""
		One times the :class:`DeliveredDose` unit of the structure dose.
		"""
		u = 1 * self.__dose
		u.value = 1
		return u

	@property
	def w_under(self):
		"""
		Underdose weight for structure's optimization objective.

		Getter returns weight normalized by structure size. Setter sets
		raw weight.

		Raises:
			TypeError: If ``w_under`` not :obj:`int` or :obj:`float`.
			ValueError: If ``w_under`` negative.
		"""
		if not positive_real_valued(self.size):
			return nan

		if isinstance(self.__w_under, (float, int)):
		    return self.__w_under / float(self.size)
		else:
			return None

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
	def w_under_raw(self):
		""" Raw underdose weight, not normalized for structure size. """
		return self.__w_under

	@property
	def w_over(self):
		"""
		Overdose weight for structure's optimization objective.

		Getter returns weight normalized by structure size. Setter sets
		raw weight.

		Raises:
			TypeError: If ``w_over`` not :obj:`int` or :obj:`float`.
			ValueError: If ``w_over`` negative.
		"""
		if not positive_real_valued(self.size):
			return nan

		if isinstance(self.__w_over, (float, int)):
			return self.__w_over / float(self.size)
		else:
			return None

	@w_over.setter
	def w_over(self, weight):
		if isinstance(weight, (int, float)):
			self.__w_over = max(0., float(weight))
			if weight < 0:
				raise ValueError('negative objective weights not allowed')
		else:
			raise TypeError('argument "weight" must be a float >= 0')

	@property
	def w_over_raw(self):
		""" Raw overdose weight, not normalized for structure size. """
		return self.__w_over

	def calculate_dose(self, beam_intensities):
		""" Alias for :meth:`Structure.calc_y`. """
		self.calc_y(beam_intensities)

	def calc_y(self, x):
		"""
		Calculate voxel doses as :attr:`Structure.y` = :attr:`Structure.A` * ``x``.

		Arguments:
			x: Vector-like input of beam intensities.

		Returns:
			None
		"""

		# calculate dose from input vector x:
		# 	y = Ax
		x = vec(x)
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
		""" Vector of structure's voxel doses. """
		return self.__y

	@property
	def mean_dose(self):
		""" Average dose to structure's voxels. """
		return self.__y_mean * self.dose_unit

	@property
	def min_dose(self):
		""" Minimum dose to structure's voxels. """
		if self.dvh is None:
			return nan * Gy
		return self.dvh.min_dose * self.dose_unit

	@property
	def max_dose(self):
		""" Maximum dose to structure's voxels. """
		if self.dvh is None:
			return nan * Gy
		return self.dvh.max_dose * self.dose_unit

	def satisfies(self, constraint):
		"""
		Test whether structure's voxel doses satisfy ``constraint``.

		Arguments:
			constraint (:class:`Constraint`): Dose constraint to test
				against structure's voxel doses.

		Returns:
			:obj:`bool`: ``True`` if structure's voxel doses conform to
			the queried	constraint.

		Raises:
			TypeError: If ``constraint`` not of type :class:`Constraint`.
			ValueError: If :attr:`Structure.dvh` not initialized or not
				populated with dose data.
		"""
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

	def plotting_data(self, constraints_only=False, maxlength=None):
		"""
		Dictionary of :mod:`matplotlib`-compatible plotting data.

		Data includes DVH curve, constraints, and prescribed dose.

		Args:
			constraints_only (:obj:`bool`, optional): If ``True``,
				return only the constraints associated with the
				structure.
			maxlength (:obj:`int`, optional): If specified, re-sample
				the structure's DVH plotting data to have a maximum
				series length of ``maxlength``.
	 	"""
	 	if constraints_only:
	 		return self.constraints.plotting_data
		else:
			return {'curve': self.dvh.resample(maxlength).plotting_data,
					'constraints': self.constraints.plotting_data,
					'rx': self.dose_rx.value,
					'target': self.is_target}

	@property
	def __header_string(self):
		""" Header string, comprising structure name and label. """
		out = '\nStructure: '
		if self.name != '':
			out += '{}'.format(self.name)
		else:
			out += '<unnamed>'
		out += ' (label = {})\n'.format(self.label)
		return out

	@property
	def objective(self):
		""" Dictionary of objective data. """
		return {
				'is_target': self.is_target,
				'dose': self.dose,
				'weight_under': self.__w_under,
				'weight_over': self.__w_over
				}

	@property
	def __obj_string(self):
		""" String of objective data. """
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
		""" String of constraints attached to :class:`Structure`. """
		out = ''
		for key in self.constraints.items:
			out += self.constraints[key].__str__()
			out += '\n'
		return out

	def summary(self, percentiles=[2, 25, 75, 98]):
		"""
		Dictionary summarizing dose statistics.

		Arguments:
			percentiles (:obj:`list`, optional): Percentile levels at
				which to query the structure dose. If not provided, will
				query doses at default percentile levels of 2%, 25%, 75%
				and 98%.

		Returns:
			:obj:`dict`: Dictionary of doses at requested percentiles,
			plus mean, minimum and maximum voxel doses.
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
		"""
		String summarizing dose statistics.

		Includes MEAN, MIN, and MAX doses, as well as doses at several
		default percentiles: 98%, 75%, 25%, 2%
		"""
		summary = self.summary()
		hdr = str(6 * '{!s:^10}|' + '{!s:^10}\n').format(
						'mean', 'min', 'max', 'D98', 'D75', 'D25', 'D2')
		vals = str(6 * '{!s:^10}|' + '{!s:^10}\n').format(
				summary['mean'], summary['min'], summary['max'],
				summary['D98'], summary['D75'], summary['D25'], summary['D2'])
		return hdr + vals

	@property
	def objective_string(self):
		""" String of structure header and objectives """
		return self.__header_string + self.__obj_string

	@property
	def constraints_string(self):
		""" String of structure header and constraints """
		return self.__header_string + self.__constr_string

	@property
	def summary_string(self):
		""" String of structure header and dose summary """
		return self.__header_string + self.__summary_string

	def __str__(self):
		""" String of structure header, objectives, and constraints """
		return self.__header_string + self.__obj_string + self.__constr_string