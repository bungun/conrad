"""
Defines :class:`Constraint` base class, along with specializations
:class:`MaxConstraint`, :class:`MinConstraint`, :class:`MeanConstraint`
and :class:`PercentileConstranint`. Also function :func:`D` for
instantiating constraints with syntax used by clinicians, e.g.::

	D('max') < 30 * Gy
	D('min') > 20 * Gy
	D('mean') > 25 * Gy
	D(90) > 22 * Gy
	D(5) < 29 * Gy

Atrributes:
	RELOPS (:class:`__ConstraintRelops`): Defines constants
		``RELOPS.GEQ``, ``RELOPS.LEQ``, and ``RELOPS.INDEFINITE`` for
		categorizing inequality constraint directions.
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

import time
import hashlib
import numpy as np

from conrad.defs import vec
from conrad.physics.units import DeliveredDose, cGy, Gy, Percent, Volume, cm3

class __ConstraintRelops(object):
	""" Define string constants for inequality constraint types. """
	GEQ = '>='
	LEQ = '<='
	INDEFINITE = '<>'

RELOPS = __ConstraintRelops()

class Constraint(object):
	"""
	Base class for dose constraints.

	The :class:`MinConstraint`, :class:`MaxConstraint`,
	:class:`MeanConstraint` and :class:`PercentileConstraint` types all
	inherit from :class:`Constraint`. This class defines all the basic
	getters and setters for constraint properties such as the type of
	threshold, constraint direction (relop) and dose bound, as well as
	other shared properties such as slack and dual values relevant to
	the :class:`Constraint` object's role in treatment plan optimization
	problems.

	"""
	def __init__(self):
		"""
		Initialize an empty, undifferentiated dose constraint.

		Arguments:
			None
		"""
		self.__dose = np.nan * Gy
		self.__threshold = None
		self.__relop = RELOPS.INDEFINITE
		self.__slack = 0.
		self.__priority = 1
		self.__prescription_dose = np.nan * Gy
		self.__dual = np.nan

	@property
	def resolved(self):
		""" Indicator that constraint is complete and well-formed. """
		if isinstance(self.dose, (DeliveredDose, Percent)):
			dose_resolved = self.dose.value is not np.nan
		else:
			dose_resolved = False
		relop_resolved = self.relop != RELOPS.INDEFINITE
		threshold_resolved = self.threshold is not None
		return dose_resolved and relop_resolved and threshold_resolved

	@property
	def threshold(self):
		""" Constraint threshold---percentile, min, max or mean. """
		return self.__threshold

	@threshold.setter
	def threshold(self, threshold):
		self.__threshold = threshold

	@property
	def relop(self):
		"""
		Constraint relop (i.e., sense of inequality).

		Should be one of ``<``, ``>``, ``<=``, or ``>=``.

		The setter method does not differentiate between strict and
		non-strict inequalities (i.e., ``<`` versus ``<=``), but both
		syntaxes are allowed for convenience.

		Raises:
			Value Error: If user tries to build a maximum dose
				constraint with a lower dose bound or a minimum dose
				constraint with an upper dose bound, or if ``relop`` is
				not one of the expected string values.
		"""
		return self.__relop

	@relop.setter
	def relop(self, relop):
		if isinstance(self.__threshold, str):
			if self.__threshold == 'max' and '>' in relop:
				raise ValueError('constraint of form Dmax > x Gy not allowed, '
								'please rephrase')
			if self.__threshold == 'min' and '<' in relop:
				raise ValueError('constraint of form Dmin < x Gy not allowed, '
								'please rephrase')
		if relop in ('<', '<='):
			self.__relop = RELOPS.LEQ
		elif relop in ('>', '>='):
			self.__relop = RELOPS.GEQ
		else:
			raise ValueError(
					'argument "relop" must be one of ("<", "<=", ">", ">=")')

	@property
	def upper(self):
		"""
		Indicator of upper dose constraint (or, 'less than' inequality).

		Arguments:
			None

		Returns:
			:obj:`bool`: ``True`` if constraint of type "D(threshold) <
			dose".

		Raises:
			ValueError: If :attr:`Constraint.relop` is not set.
		"""
		if self.__relop == RELOPS.INDEFINITE:
			raise ValueError('{} object field "relop" not initialized, '
							 'identity as upper/lower constraint undefined'
							 ''.format(Constraint))
		return self.__relop == RELOPS.LEQ

	@property
	def dose(self):
		"""
		Dose bound for constraint.

		Getter returns dose in absolute terms (i.e.,
		:class:`DeliveredDose` units.)

		Setter accepts dose in absolute or relative terms. That is,
		dose may be provided provided in units of :class:`Percent` or in
		units of :class:`DeliveredDose`, such as
		:class:`~conrad.physics.units.Gray`.

		Raises:
			TypeError: If ``dose`` not of allowed types.
		"""
		if isinstance(self.__dose, Percent):
			if self.__prescription_dose.value is np.nan:
				return self.__prescription_dose
			else:
				return self.__dose * self.__prescription_dose
		else:
			return self.__dose

	@dose.setter
	def dose(self, dose):
		if isinstance(dose, (DeliveredDose, Percent)):
			self.__dose = dose
		else:
			raise TypeError('argument "dose" must be of type {}, {} or '
							'{}'.format(Percent, type(Gy), type(cGy)))

	@property
	def rx_dose(self):
		"""
		Prescription dose associated with constraint.

		This property is optional, but required when the
		:attr:`Constraint.dose` is phrased in relative terms (i.e., of
		type :class:`Percent`). It provides the numerical basis on which
		to interpret the relative value of :attr:`Constraint.dose`.

		Raises:
			TypeError: If ``rx_dose`` is not of type
				:class:`DeliveredDose`, e.g.,
				:class:`~conrad.physics.units.Gray` or
				:class:`~conrad.physics.units.centiGray`.
		"""
		return self.__prescription_dose

	@rx_dose.setter
	def rx_dose(self, rx_dose):
		if isinstance(rx_dose, DeliveredDose):
			self.__prescription_dose = rx_dose
		else:
			raise TypeError('argument "rx_dose" must be of type {}'.format(
							type(Gy), type(cGy)))

	@property
	def active(self):
		""" True if constraint active in most recent plan that used it."""
		return self.__dual > 0.

	@property
	def dual_value(self):
		"""
		Value of dual variable associated with constraint.

		This property is intended to reflect information about the state
		of the :class:`Constraint` in the context of the most recent run
		of an optimization problem that it was used in. Accordingly, it
		is to be managed by some client(s) of the :class:`Constraint`
		and not the object itself.

		In particular, this property is meant to hold the value of the
		dual variable associated with the dose constraint in some
		solver's representation of an optimization problem, and the
		value should be that attained at the conclusion of a solver run.
		"""
		return self.__dual

	@dual_value.setter
	def dual_value(self, dual_value):
		self.__dual = float(dual_value)

	@property
	def slack(self):
		"""
		Value of slack variable associated with constraint.

		This property is intended to reflect information about the state
		of the :class:`Constraint` in the context of the most recent run
		of an optimization problem that it was used in. Accordingly, it
		is to be managed by some client(s) of the :class:`Constraint`
		and not the object itself.

		In particular, this property is meant to hold the value of the
		slack variable associated with the dose constraint in some
		solver's representation of an optimization problem, and the
		value should be that attained at the conclusion of a solver run.

		Raises:
			TypeError: If ``slack`` is not an :obj:`int` or :obj:`float`.
			ValueError: If ``slack`` is negative.
		"""
		return self.__slack

	@slack.setter
	def slack(self, slack):
		# TODO: Remove this once we find bug that is turning slack into 1-element array.
		if isinstance(slack, np.ndarray) and slack.size == 1:
			slack = np.asscalar(slack)
		
		if isinstance(slack, (int, float)):
			if slack < 0:
				raise ValueError('argument "slack" must be nonnegative '
								 'by convention')
			self.__slack = max(0., float(slack))
		else:
			raise TypeError('argument "slack" must be of type {} with '
							'value >= 0'.format(float))

	@property
	def dose_achieved(self):
		""" Constraint dose +/- slack. """
		sign = +1 if self.upper else -1
		return self.dose + sign * self.__slack

	@property
	def priority(self):
		"""
		Constraint priority.

		Priority is one of {``0``, ``1``, ``2``, ``3``}. Constraint
		priorities are used when incorporating a :class:`Constraint` in
		an optimization problem with slack allowed.

		Priority ``0`` indicates that the constraint should be enforced
		strictly even when the overall problem formulation permits
		dose constraint slacks.

		The remaining values (``1``, ``2``, and ``3``) represent ranked
		tiers; slacks are permitted and penalized according to the
		priority: the slack variable for a ``Priority 1`` constraint is
		penalizeed more heavily than that of a ``Priority 2``
		constraint, which is in turn penalized more heavily than the
		slack variable associated with a ``Priority 3`` constraint. This
		mechanism allows users to encourage some constraints to be met
		more closely than others, even when slack is allowed for all of
		them.

		Raises:
			TypeError: If ``priority`` not an :obj:`int`:.
			ValueError: If ``priority`` not in {``0``, ``1``, ``2``,
				``3``}.
		"""
		return self.__priority

	@priority.setter
	def priority(self, priority):
		if isinstance(priority, (int, float)):
			self.__priority = max(0, min(3, int(priority)))
			if priority < 0:
				raise ValueError('argument "priority" cannot be negative. '
								 'allowed values: {0, 1, 2, 3}')
			elif priority > 3:
				raise ValueError('argument "priority" cannot be > 3. allowed '
								 'values: {0, 1, 2, 3}')
		else:
			raise TypeError('argument "priority" must be an integer between 0 '
							'and 3')

	@property
	def symbol(self):
		""" Strict inequality :obj:`str` of :attr:`Constraint.relop`. """
		return self.__relop.replace('=', '')

	def __lt__(self, other):
		"""
		Overload operator <.

		Enable :attr:`Constraint.dose` and :attr:`Constraint.relop` to
		be set via syntax 'constraint < dose'.

		Arguments:
			other: Value that :attr:`Constraint.dose` will be set to.

		Returns:
			:class:`Constraint`: Updated version of this object.
		"""
		self.relop = RELOPS.LEQ
		self.dose = other
		return self

	def __le__(self, other):
		""" Reroute operator <= to operator < """
		return self.__lt__(other)

	def __gt__(self, other):
		"""
		Overload operator >.

		Enable :attr:`Constraint.dose` and :attr:`Constraint.relop` to
		be set via syntax 'constraint > dose'.

		Arguments:
			other: Value that :attr:`Constraint.dose` will be set to.

		Returns:
			:class:`Constraint`: Updated version of this object.
		"""
		self.relop = RELOPS.GEQ
		self.dose = other
		return self

	def __ge__(self, other):
		""" Reroute operator >= to operator > """
		return self.__gt__(other)

	def __eq__(self, other):
		"""
		Overload operator ==

		Define comparison between constraints by comparing their relops,
		doses and thresholds.

		Arguments:
			other (:class:'Constraint'): Value to be compared.

		Returns:
			:obj:bool: ``True`` if compared constraints are equivalent.

		Raises:
			TypeError: If ``other`` is not a :class:`Constraint`.
		"""
		if not isinstance(other, Constraint):
			raise TypeError('equality comparison for {} object only '
							'defined when both operands of type {}'
							''.format(Constraint, Constraint))

		# filter out mixed comparisons between percentile vs. mean/min/max
		if type(self.threshold) != type(other.threshold):
			return False

		equal = self.dose == other.dose
		equal &= self.relop == other.relop
		equal &= self.threshold == other.threshold
		if isinstance(self.dose, Percent):
			equal &= self.rx_dose == other.rx_dose
		return equal

	def __str__(self):
		"""
		Stringify :class:`Constraint` as 'D{threshold} {<= or >=} {dose}'
		"""
		thresh = self.__threshold
		thresh = str(thresh.value) if isinstance(thresh, Percent) else thresh
		return str('D{} {} {}'.format(thresh, self.__relop, self.__dose))

class PercentileConstraint(Constraint):
	"""
	Percentile, i.e. dose-volume, or partial dose constraint.

	Allow for dose bounds to be applied to a certain fraction of a
	structure involved in treatment planning. For instance, a lower
	constraint,

	>>>	# D80 > 60 Gy,

	requires at least 80% of the voxels in a structure must receive 60
	Grays or greater, and an upper constraint,

	>>>	# D25 < 5 Gy,

	requires no more than 25% of the voxels in a structure to receive
	25 Grays or greater.

	Extend base class :class:`Constraint`, recast
	:attr:`Constraint.threshold` as
	:attr:`PercentileConstraint.percentile`.

	"""
	def __init__(self, percentile=None, relop=None, dose=None):
		"""
		Initialize and optionally populate a percentile constraint.

		Arguments:
			percentile (optional): Percentile threshold. Expected to be
				compatible with :attr:`PercentileConstraint.percentile`
				setter.
			relop (optional): Sense of inequality. Expected to be
				compatible with :attr:`Constraint.relop` setter.
			dose (optional): Dose bound. Expected to be compatible with
				:attr:`Constraint.dose` setter.
		"""
		Constraint.__init__(self)
		if relop is not None:
			self.relop = relop
		if dose is not None:
			self.dose = dose
		if percentile is not None:
			self.percentile = percentile

	@property
	def percentile(self):
		"""
		Percentile threshold in interval (``0``, ``100``).

		Raises:
			TypeError: If ``percentile`` is not :obj:`int`,
				:obj:`float`, or :class:`Percent`.
		"""
		return self.threshold

	@percentile.setter
	def percentile(self, percentile):
		if isinstance(percentile, (int, float)):
			self.threshold = min(100., max(0., float(percentile))) * Percent()
		elif isinstance(percentile, Percent):
			self.threshold = percentile
		else:
			raise TypeError('argument "percentile" must be of type {}, {} or '
					  '{}'.format(int, float, Percent))

	@property
	def plotting_data(self):
		""" Dictionary of :mod:`matplotlib`-compatible data. """
		return {
				'type': 'percentile',
				'percentile' : 2 * [self.percentile.value],
				'dose' :[self.dose.value, self.dose_achieved.value],
				'symbol' : self.symbol
				}

	def get_maxmargin_fulfillers(self, y, had_slack=False):
		r"""
		Get indices to values of ``y`` deepest in feasible set.

		In particular, given ``len(y)``, if ``m`` voxels are
		required to respect this :class:`PercentileConstraint` exactly,
		``y`` is assumed to contain at least ``m`` entries that respect
		the constraint (for instance, ``y`` is generated by a convex
		program that includes a convex restriction of the dose
		constraint).

		Procedure.

		.. math::
		   :nowrap:

		   \begin{array}{rl}
		   \mathbf{0.} & \mbox{Define} \\
		   	& p = \mbox{percent non-violating} \cdot \mbox{structure size}
			    = \mbox{percent non-violating} \cdot \mathbf{len}(y) \\
		   \mathbf{1.} & \mbox{Get margins: } y - \mbox{dose bound}. \\
		   \mathbf{2.} & \mbox{Sort margin indices by margin values.} \\
		   \mathbf{3.} & \mbox{If upper constraint, return indices of
		   $p$ most negative entries}. \\
		   \mathbf{4.} & \mbox{If lower constraint, return indices of
		   $p$ most positive entries}. \\
		   \end{array}

		Arguments:
			y: Vector-like input data of length ``m``.
			had_slack (:obj:`bool`, optional): Define margin relative to
				slack-modulated dose value instead of the base dose
				value of this :class:`PercentileConstraint`.

		Returns:
			:class:`numpy.ndarray`: Vector of indices that yield the
			``p`` entries of ``y`` that fulfill this
			:class:`PercentileConstraint` with the greatest margin.
		"""
		fraction = self.percentile.fraction
		non_viol = (1 - fraction) if self.upper else fraction
		n_returned = int(np.ceil(non_viol * len(y)))

		start = 0 if self.upper else -n_returned
		end = n_returned if self.upper else None
		dose = self.dose_achieved.value if had_slack else self.dose.value
		return (vec(y) - dose).argsort()[start:end]

class MeanConstraint(Constraint):
	"""
	Mean dose constraint.

	Extend base class :class:`Constraint`. Express an upper or lower
	bound on the mean dose to a structure.
	"""
	def __init__(self, relop=None, dose=None):
		"""
		Initialize and optionally populate a mean dose constraint.

		Arguments:
			relop (optional): Sense of inequality. Expected to be
				compatible with :attr:`Constraint.relop` setter.
			dose (optional): Dose bound. Expected to be compatible with
				:attr:`Constraint.dose` setter.
		"""
		Constraint.__init__(self)
		if relop is not None:
			self.relop = relop
		if dose is not None:
			self.dose = dose
		self.threshold = 'mean'

	@property
	def plotting_data(self):
		""" Dictionary of :mod:`matplotlib`-compatible data. """
		return {'type': 'mean',
			'dose' : [self.dose.value, self.dose_achieved.value],
			'symbol' : self.symbol}

class MinConstraint(Constraint):
	"""
	Minimum dose constraint.

	Extend base class :class:`Constraint`. Express a lower bound on the
	minimum dose to a structure.
	"""
	def __init__(self, relop=None, dose=None):
		"""
		Initialize and optionally populate a minimum dose constraint.

		Arguments:
			relop (optional): Sense of inequality. Expected to be
				compatible with :attr:`Constraint.relop` setter.
			dose (optional): Dose bound. Expected to be compatible with
				:attr:`Constraint.dose` setter.
		"""
		Constraint.__init__(self)
		if relop is not None:
			self.relop = relop
		if dose is not None:
			self.dose = dose
		self.threshold = 'min'

	@property
	def plotting_data(self):
		""" Dictionary of :mod:`matplotlib`-compatible data. """
		return {'type': 'min',
			'dose' : [self.dose.value, self.dose_achieved.value],
			'symbol' : self.symbol}

class MaxConstraint(Constraint):
	"""
	Maximum dose constraint.

	Extend base class :class:`Constraint`. Express an upper bound on the
	maximum dose to a structure.
	"""
	def __init__(self, relop=None, dose=None):
		"""
		Initialize and optionally populate a maximum dose constraint.

		Arguments:
			relop (optional): Sense of inequality. Expected to be
				compatible with :attr:`Constraint.relop` setter.
			dose (optional): Dose bound. Expected to be compatible with
				:attr:`Constraint.dose` setter.
		"""
		Constraint.__init__(self)
		if relop is not None:
			self.relop = relop
		if dose is not None:
			self.dose = dose
		self.threshold = 'max'

	@property
	def plotting_data(self):
		""" Dictionary of :mod:`matplotlib`-compatible data. """
		return {'type': 'max',
			'dose' :[self.dose.value, self.dose_achieved.value],
			'symbol' : self.symbol}

def D(threshold, relop=None, dose=None):
	"""
	Utility for constructing dose constraints with clinical syntax.

	Arguments:
		threshold: Specify type of dose constraint; if real-valued or
			of type :class:`Percent`, parsed as a percentile constraint.
			If string-valued, tentatively interpreted as a mean,
			minimum, or maximum type dose constraint.
		relop (optional): Sense of inequality. Expected to be compatible
			with :attr:`Constraint.relop` setter.
		dose (optional): Dose bound. Expected to be compatible with
			:attr:`Constraint.dose` setter.

	Returns:
		:class:`Constraint`: Return type depends on argument
		``threshold``.

	Raises:
		ValueError: If ``threshold`` does not conform to expected types
			or formats.

	Examples:
		>>> D('mean') > 30 * Gy
		>>> D('min') > 10 * Gy
		>>> D('max') < 5 * Gy
		>>> D(30) < 4 * Gy
		>>> D(90) > 47 * Gy
	"""
	if isinstance(threshold, str):
		if threshold in ('mean', 'Mean'):
			return MeanConstraint(relop=relop, dose=dose)
		elif threshold in ('min', 'Min', 'minimum', 'Minimum') or threshold == 100:
			return MinConstraint(relop=relop, dose=dose)
		elif threshold in ('max', 'Max', 'maximum', 'Maximum') or threshold == 0:
			return MaxConstraint(relop=relop, dose=dose)
	elif isinstance(threshold, (int, float, Percent)):
		return PercentileConstraint(percentile=threshold, relop=relop, dose=dose)
	raise ValueError('constraint unparsable as phrased')

class AbsoluteVolumeConstraint(Constraint):
	def __init__(self, dose=None, relop=None, volume_threshold=None,
				 structure_volume=None):
		Constraint.__init__(self)
		self.__constraint_volume = None
		self.__total_volume = None

		if dose is not None:
			self.dose = dose
		if relop is not None:
			self.relop = relop
		if volume_threshold is not None:
			self.volume = volume_threshold
		if structure_volume is not None:
			self.total_volume = structure_volume

	# overload property Constraint.resolved to always be false:
	# force conversion to PercentileConstraint for planning
	@property
	def resolved(self):
		raise ValueError('{} is unresolvable by convention: please convert '
						 'to {} using built-in conversion'
						 ''.format(
						 		AbsoluteVolumeConstraint, PercentileConstraint))

	def __lt__(self, other):
		"""
		Overload operator <.

		Enable :attr:`AbsoluteVolumeConstraint.volume` and
		:attr:`Constraint.relop` to
		be set via syntax 'constraint < volume'.

		Arguments:
			other: Value that :attr:`AbsoluteVolumeConstraint.volume` will be set to.

		Returns:
			:class:`AbsoluteVolumeConstraint`: Updated version of this object.
		"""
		self.relop = RELOPS.LEQ
		self.volume = other
		return self

	def __gt__(self, other):
		"""
		Overload operator >.

		Enable :attr:`AbsoluteVolumeConstraint.volume` and :attr:`Constraint.relop` to
		be set via syntax 'constraint > volume'.

		Arguments:
			other: Value that :attr:`AbsoluteVolumeConstraint.dose` will be set to.

		Returns:
			:class:`AbsoluteVolumeConstraint`: Updated version of this object.
		"""
		self.relop = RELOPS.GEQ
		self.volume = other
		return self

	@property
	def volume(self):
		return self.__constraint_volume

	@volume.setter
	def volume(self, volume):
		if not isinstance(volume, Volume):
			raise TypeError(
					'argument `volume` must be of type {}'
					''.format(Volume))
		self.__constraint_volume = volume

	@property
	def total_volume(self):
		return self.__total_volume

	@total_volume.setter
	def total_volume(self, volume):
		if not isinstance(volume, Volume):
			raise TypeError(
					'argument `volume` must be of type {}'
					''.format(Volume))
		self.__total_volume = volume

	def to_percentile_constraint(self, structure_volume=None):
		if self.total_volume is None:
			if structure_volume is None:
				raise ValueError(
						'to convert to percentile constraint, '
						'`AbsoluteVolumeConstraint.total_volume` must '
						'be set prior to method call, or argument '
						'`structure_volume` must be provided')
			self.total_volume = structure_volume

		if self.volume is None:
			raise ValueError(
					'constraint absolute volume unspecified, cannot '
					'use structure volume to convert to relative '
					'volume constraint')

		ratio = self.volume.to_cm3.value / self.total_volume.to_cm3.value

		if ratio > 1:
			raise ValueError('conversion from {} to {} failed.\n'
							 'cannot form a {} with a percentile greater than '
							 '100%.\nRequested: {:0.1f}\n'
							 '(constrained volume / total volume = {}/{})'
							 ''.format(AbsoluteVolumeConstraint,
							 PercentileConstraint, PercentileConstraint,
							 100 * ratio, self.volume.to_cm3,
							 self.total_volume.to_cm3))
		elif ratio == 1 or ratio == 0:
			raise ValueError('conversion from {} to {} failed.\n'
							 'constrained volume / total volume = {}/{})\n'
							 'rephrase as min dose or max dose constraint'
							 ''.format(AbsoluteVolumeConstraint,
							 PercentileConstraint, self.volume.to_cm3,
							 self.total_volume.to_cm3))

		return PercentileConstraint(
				100 * ratio * Percent(), self.relop, self.dose)


class GenericVolumeConstraint(Constraint):
	def __init__(self, dose=None, relop=None, volume_threshold=None):
		Constraint.__init__(self)
		self.__constraint_volume = None

		if dose is not None:
			self.dose = dose
		if relop is not None:
			self.relop = relop
		if volume_threshold is not None:
			self.volume = volume_threshold

	@property
	def volume(self):
		return self.__constraint_volume

	@volume.setter
	def volume(self, volume):
		if isinstance(volume, Percent):
			self.threshold = volume
		elif isinstance(volume, Volume):
			self.__constraint_volume = volume
		else:
			raise TypeError(
				'volume threshold for a GenericVolume constraint must '
				'either be a relative volume (expressed as '
				':class:`Percent`) or an absolute volume (expressed as '
				'type :class:`Volume`')

	# overload property Constraint.resolved to always be false:
	# force conversion to PercentileConstraint for planning
	@property
	def resolved(self):
		raise ValueError('{} is unresolvable by convention: please convert '
						 'to {} or {} using built-in conversion'
						 ''.format(
						 		GenericVolumeConstraint,
						 		AbsoluteVolumeConstraint,
						 		PercentileConstraint))

	def __lt__(self, other):
		dose_arg = self._Constraint__dose
		if isinstance(other, Percent):
			return PercentileConstraint(other, RELOPS.LEQ, dose_arg)
		elif isinstance(other, Volume):
			return AbsoluteVolumeConstraint(dose_arg, RELOPS.LEQ, other)
		else:
			self.relop = RELOPS.LEQ
			self.volume = other

	def __gt__(self, other):
		dose_arg = self._Constraint__dose
		if isinstance(other, Percent):
			return PercentileConstraint(other, RELOPS.GEQ, dose_arg)
		elif isinstance(other, Volume):
			return AbsoluteVolumeConstraint(dose_arg, RELOPS.GEQ, other)
		else:
			self.relop = RELOPS.GEQ
			self.volume = other

	def specialize(self):
		relop_arg = self.relop if self.relop != RELOPS.INDEFINITE else None
		dose_arg = self._Constraint__dose
		if self.threshold is not None:
			return PercentileConstraint(self.threshold, relop_arg, dose_arg)
		elif self.volume is not None:
			return AbsoluteVolumeConstraint(dose_arg, relop_arg, self.volume)
		else:
			return self

def V(dose, relop=None, threshold=None):
	return GenericVolumeConstraint(dose, relop, threshold).specialize()
	return c.specialize()
