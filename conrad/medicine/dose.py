"""
Define :class:`Constraint` base class, along with specializations
:class:`MaxConstraint`, :class:`MinConstraint`, :class:`MeanConstraint`
and :class:`PercentileConstranint`. Also define :class:`ConstraintList`
container and function :func:`D` for instantiating constraints with a
syntax used by clinicians, e.g.::

	D('max') < 30 * Gy
	D('min') > 20 * Gy
	D('mean') > 25 * Gy
	D(90) > 22 * Gy
	D(5) < 29 * Gy

Also define the :class:`DVH` (dose volume histogram) object for
converting structure dose vectors to plottable DVH data sets.

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

from time import clock
from hashlib import sha1
from numpy import zeros, linspace, nan, ceil

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
		self.__dose = nan
		self.__threshold = None
		self.__relop = RELOPS.INDEFINITE
		self.__slack = 0.
		self.__priority = 1
		self.__prescription_dose = nan * Gy
		self.__dual = nan

	@property
	def resolved(self):
		""" Indicator that constraint is complete and well-formed. """
		if isinstance(self.dose, (DeliveredDose, Percent)):
			dose_resolved = self.dose.value is not nan
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
			if self.__prescription_dose.value is nan:
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
		n_returned = int(ceil(non_viol * len(y)))

		start = 0 if self.upper else -n_returned
		end = n_returned if self.upper else None
		dose = self.dose_achieved.value if had_slack else self.dose.value
		return (vec(y) - dose).argsort()[start:end]


# class AbsoluteVolumeConstraint(Constraint):
# 	def __init__(self, volume=None, relop=None, dose=None):
# 		Constraint.__init__(self)
# 		if relop is not None:
# 			self.relop = relop
# 		if dose is not None:
# 			self.dose = dose
# 		if volume is not None:
# 			self.volume = volume
# 		else:
# 			self.volume = nan * cm3
# 		self.__total_volume = nan * cm3

# 	# overload property Constraint.resolved to always be false:
# 	# force conversion to PercentileConstraint for planning
# 	@property
# 	def resolved(self):
# 		raise ValueError('{} is unresolvable by convention: please convert '
# 						 'to {} using built-in conversion')

# 	@property
# 	def volume(self):
# 		return self.threshold

# 	@volume.setter
# 	def volume(self, volume):
# 		if not isinstance(volume, Volume):
# 			raise TypeError('argument "volume" must be of type {}'
# 							''.format(Volume))
# 		else:
# 			self.threshold = volume

# 	@property
# 	def total_volume(self):
# 		return self.__total_volume

# 	@total_volume.setter
# 	def total_volume(self, total_volume):
# 		if not isinstance(total_volume, Volume):
# 			raise TypeError('argument "total_volume" must be of type {}'
# 							''.format(Volume))
# 		self.__total_volume = total_volume

# 	@property
# 	def to_percentile_constraint(self, total_volume=None):
# 		if total_volume is not None:
# 			self.total_volume = total_volume
# 		if self.total_volume.value in (nan, None):
# 			raise ValueError('field "total_volume" of {} object must be set '
# 							 'for conversion to {} to be possible'.format(
# 							 AbsoluteVolumeConstraint, PercentileConstraint))
# 		fraction = self.volume.to_cm3.value / self.total_volume.to_cm3.value
# 		if fraction > 1:
# 			raise ValueError('conversion from {} to {} failed.\n'
# 							 'cannot form a {} with a percentile greater than '
# 							 '100%.\nRequested: {:0.1f}\n'
# 							 '(constrained volume / total volume = {}/{})'
# 							 ''.format(AbsoluteVolumeConstraint,
# 							 PercentileConstraint, PercentileConstraint,
# 							 100 * fraction, self.volume.to_cm3,
# 							 self.total_volume.to_cm3))
# 		elif fraction == 1 or fraction == 0:
# 			raise ValueError('conversion from {} to {} failed.\n'
# 							 'constrained volume / total volume = {}/{})\n'
# 							 'rephrase as min dose or max dose constraint'
# 							 ''.format(AbsoluteVolumeConstraint,
# 							 PercentileConstraint, self.volume.to_cm3,
# 							 self.total_volume.to_cm3))
# 		else:
# 			return PercentileConstraint(100 * fraction * Percent(), self.relop,
# 										self.dose)


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

# class VolumeAtOrAbove(object):
	# def __init__(self, dose, relop=None, threshold=None)
	#

# class AbsoluteVolumeAtOrAbove(VolumeAtOrAbove):

# class FractionalVolumeAtOrAbove(VolumeAtOrAbove):

# class UnresolvedVolumeAtOrAbove(object):


# class

# class UnresolvedVolumeConstraint(Constraint):
# 	def __init__(self, dose=None, relop=None, threshold=None):
# 		Constraint.__init__(self)
# 		if dose is not None:
# 			self.dose = dose
# 		if relop is not None:
# 			self.relop = relop
# 		if threshold is not None:
# 			self.threshold = threshold

# 	def __le__(self, other):
# 		if isinstance(other, Percent):
# 			return D(other) >= self.dose

# 	def __lt__(self, other):
# 		return self.__le__(other)

# 	def __ge__(self, other):
# 		if isinstance(other, Percent):
# 			return D(other) <= self.dose

# 	def __gt__(self, other):
# 		return self.__ge__(other)


# def V(dose, relop=None, threshold=None):
# 	c = UnresolvedVolumeConstraint(dose, relop, threshold)
# 	return c

class ConstraintList(object):
	"""
	Container for :class:`Constraint` objects.

	Attributes:
		items (:obj:`dict`): Dictionary of constraints in container,
			keyed by hashed values generated upon addition of constraint
			to container.
		last_key: Key generated upon most recent addition of a
			constraint to the container.
	"""
	def __init__(self):
		"""
		Initialize bare :class:`ConstraintList` container.

		Arguments:
			None
		"""
		self.items = {}
		self.last_key = None

	@staticmethod
	def __keygen(constraint):
		"""
		Build unique identifier for ``constraint``.

		Hash current time and constraint properties (dose, relop,
		threshold) to generate unique identifier, take first seven
		characters as a key with low probability of collision.

		Arguments:
			constraint (:class:`Constraint`): Dose constraint to be
				assigned a key.

		Returns:
			:obj:`str`: Seven character key.
		"""
		return sha1(str(
				str(clock()) +
				str(constraint.dose) +
				str(constraint.threshold) +
				str(constraint.relop)
			).encode('utf-8')).hexdigest()[:7]

	def __getitem__(self, key):
		""" Overload operator []. """
		return self.items[key]

	def __iter__(self):
		""" Python3-compatible iterator implementation. """
		return self.items.__iter__()

	def __next__(self):
		""" Python3-compatible iterator implementation. """
		return self.items.__next__()

	def next(self):
		""" Python2-compatible iterator implementation. """
		return self.items.next()

	def iteritems(self):
		""" Python2-compatible iterator implementation. """
		return self.items.items()

	def itervalues(self):
		""" Python2-compatible iterator implementation. """
		return self.items.values()

	def __iadd__(self, other):
		"""
		Overload operator +=.

		Enable syntax :class:`ConstraintList` += :class:`Constraint`.

		Arguments:
			other: Singleton, or iterable collection of
				:class:`Constraint` objects to append to this
				:class:`ConstraintList`.

		Returns:
			:class:`ConstraintList`: Updated version of this object.

		Raises:
			TypeError: If ``other`` is not a :class:`Constraint` or
				iterable collection of constraints.
		"""
		if isinstance(other, Constraint):
			key = self.__keygen(other)
			self.items[key] =  other
			self.last_key = key
			return self
		elif isinstance(other, (list, tuple)):
			for constr in other:
				self += constr
			return self
		elif isinstance(other, ConstraintList):
			self += other.items
			return self
		elif isinstance(other, dict):
			for constr in other.values():
				self += constr
			return self
		else:
			raise TypeError('argument must be of type {} or {}'.format(
							Constraint, ConstraintList))

	def __isub__(self, other):
		"""
		Overload operator -=.

		Enables syntaxes
			:class:`ConstraintList` -= :class:`Constraint`, and
			:class:`ConstraintList` -= ``key``.

		Remove ``other`` from this :class:`ConstraintList` if it is a
		key with a corresponding :class:`Constraint`, *or* if it is a
		:class:`Constraint` for which an exactly equivalent
		:class:`Constraint` is found in the list.

		Arguments:
			other: :class:`Constraint` or key to a :class:`Constraint`
				to be removed from this :class:`ConstraintList`.

		Returns:
			:class:`ConstraintList`: Updated version of this object.
		"""
		if isinstance(other, Constraint):
			for key, constr in self.items.items():
				if other == constr:
					del self.items[key]
					return self
		else:
			if other in self.items:
				del self.items[other]
				return self

	@property
	def size(self):
		""" Number of constraints in list. """
		return len(self.items)

	@property
	def keys(self):
		""" Keys of constraints in list. """
		return self.items.keys()

	@property
	def list(self):
		"""
		:obj:`list` of :class:`Constraint` objects in :class:`ConstraintList`.
		"""
		return self.items.values()


	@property
	def mean_only(self):
		""" ``True`` if list exclusively contains mean constraints. """
		meantest = lambda c : isinstance(c, MeanConstraint)
		if self.size == 0:
			return True
		else:
			return all(listmap(meantest, self.itervalues()))

	def contains(self, constr):
		"""
		Test whether search :class:`Constraint` exists in this :class:`ConstraintList`.

		Arguments:
			constr (:class:`Constraint`): Search term.

		Returns:
			:obj:`bool`: ``True`` if a :class:`Constraint` equivalent to
			``constr`` found in this :class:`ConstraintList`.
		"""
		return constr in self.items.values()

	def clear(self):
		"""
		Clear constraints from :class:`ConstraintList`.

		Arguments:
			None

		Returns:
			None
		"""
		self.items = {}

	@property
	def plotting_data(self):
		"""
		List of :mod:`matplotlib`-compatible data for all constraints.
		"""
		return [(key, dc.plotting_data) for key, dc in self.items.items()]

	def __str__(self):
		""" Stringify list by concatenating strings of each constraint. """
		out = '(keys):\t (constraints)\n'
		for key, constr in self.items.items():
			out += key + ':\t' + str(constr) + '\n'
		return out

class DVH(object):
	"""
	Representation of a dose volume histogram.

	Given a vector of doses, the :class:`DVH` object generates the
	corresponding dose volume histogram (DVH).

	A DVH is associated with a planning structure, which will have a
	finite volume or number of voxels. A DVH curve (or graph) is a set
	of points :math:`(d, p)`---with :math:`p` in the interval
	:math:`[0, 100]`--where for each dose level :math:`d`, the value
	:math:`p` gives the percent of voxels in the associated structure
	receiving a radiation dose :math:`\ge d`.

	Sampling is performed if necessary to keep data series length short
	enough to be conveniently transmitted (e.g., as part of an
	interactive user interface) and plotted (e.g., with
	:mod:`matplotlib` utilities) with low latency.

	Note that the set of (dose, percentile) pairs are maintained as
	two sorted, length-matched, vectors of dose and percentile values,
	respectively.

	Attributes:
		MAX_LENGTH (:obj:`int`): Default maximum length constant to use
			when constructing and possibly sampling DVH cures.
	"""
	MAX_LENGTH = 1000

	def __init__(self, n_voxels, maxlength=MAX_LENGTH):
		"""
		Initialize :class:`DVH`.

		Set sizes of full dose data for associated structure, as well as
		a practical, maximum size to sample down to if necessary.

		Arguments:
			n_voxels (:obj:`int`): Number of voxels in the structure
				associated with this :class:`DVH`.
			maxlength (:obj:`int`, optional): Maximum series length,
				above which data will be sampled to maintain a suitably
				short representation of the DVH.

		Raises:
			ValueError: If ``n_voxels`` is not an :obj:`int` >= `1`.
		"""
		if n_voxels is None or n_voxels is nan or n_voxels < 1:
			raise ValueError('argument "n_voxels" must be an integer > 0')

		self.__dose_buffer = zeros(int(n_voxels))
		self.__stride = 1 * (n_voxels < maxlength) + int(n_voxels / maxlength)
		length = len(self.__dose_buffer[::self.__stride]) + 1
		self.__doses = zeros(length)
		self.__percentiles = zeros(length)
		self.__percentiles[0] = 100.
		self.__percentiles[1:] = linspace(100, 0, length - 1)
		self.__DATA_ENTERED = False

	@property
	def populated(self):
		""" True if DVH curve is populated. """
		return self.__DATA_ENTERED

	@property
	def data(self):
		"""
		Sorted dose values from DVH curve.

		The data provided to the setter are sorted to form the abscissa
		values for the DVH curve. If the length of the input exceeds the
		maximum data series length (as determined when the object was
		initialized), the input data is sampled.

		Raises:
			ValueError: If size of input data does not match size of
			structure associated with :class:`DVH` as specified to
			object initializer.
		"""
		return self.__doses[1:]

	@data.setter
	def data(self, y):
		if len(y) != self.__dose_buffer.size:
			raise ValueError('dimension mismatch: length of argument "y" '
							 'must be {}'.format(self.__dose_buffer.size))

		# populate dose buffer from y
		self.__dose_buffer[:] = y[:]

		# maintain sorted buffer
		self.__dose_buffer.sort()

		# sample doses from buffer
		self.__doses[1:] = self.__dose_buffer[::self.__stride]

		# flag DVH curve as populated
		self.__DATA_ENTERED = True

	@staticmethod
	def __interpolate_percentile(p1, p2, p_des):
		r"""
		Get ``alpha`` such that:

		:math: alpha * `p1` + (1-alpha) * `p2` = `p_des`.

		Arguments:
			p1 (:obj:`float`): First endpoint of interval (``p1``,
				``p2``).
			p2 (:obj:`float`): Second endpoint of interval (``p1``,
				``p2``).
			p_des (:obj:`float`): Desired percentile. Should be
				contained in interval (``p1``, ``p2``) for this method
				to be a linear interpolation.

		Returns:
			float: Value ``alpha`` such that linear combination
			``alpha`` * ``p1`` + ``(1 - alpha)`` * ``p2`` yields
			``p_des``.

		Raises:
			ValueError: If request is ill-posed because ``p1`` and
				``p2`` coincide, but ``p_des`` is a different value.
		"""

		if p1 == p2 and p_des == p1:
			return 1.
		elif p1 == p2 and p_des != p1:
			raise ValueError('arguments "p1", "p2" must be distinct to '
							 'perform interpolation.\nprovided interval: '
							 '[{},{}]\ntaget value: {}'.format(p1, p2, p_des))
		else:
			# alpha * p1 + (1 - alpha) * p2 = p_des
			# (p1 - p2) * alpha = p_des - p2
			# alpha = (p_des - p2) / (p1 - p2)
			return float(p_des - p2) / float(p1 - p2)

	def dose_at_percentile(self, percentile):
		"""
		Read off DVH curve to get dose value at ``percentile``.

		Since the :class:`DVH` object maintains the DVH curve of
		(dose, percentile) pairs as two sorted vectors, to approximate
		the dose d at percentile ``percentile``, we retrieve the
		index i that yields the nearest percentile value. The
		corresponding i'th dose is returned. When the nearest percentile
		is not within 0.5%, the two nearest neighbor percentiles and
		two nearest neighbor dose values are used to approximate the
		dose at the queried percentile by linear interpolation.

		Arguments:
			Percentile (:obj:`int`, :obj:`float` or :class:`Percent`):
				Queried percentile for which to retrieve corresponding
				dose level.

		Returns:
			Dose value from DVH curve corresponding to queried
			percentile, or :attr:`~numpy.nan` if the curve has not been
			populated with data.
		"""
		if isinstance(percentile, Percent):
			percentile = percentile.value

		if self.__doses is None: return nan

		if percentile == 100: return self.min_dose
		if percentile == 0: return self.max_dose

		# bisection retrieval of index @ percentile
		# ----------------------------------------
		u = len(self.__percentiles) - 1
		l = 1
		i = int(l + (u - l) / 2)

		# set tolerance based on bucket width
		tol = (self.__percentiles[-2] - self.__percentiles[-1]) / 2

		# get to within 0.5 of a percentile if possible
		abstol = 0.5

		while (u - l > 5):
			# percentile sorted descending
			if self.__percentiles[i] > percentile:
				l = i
			else:
				u = i
			i = int(l + (u - l) / 2)

		# break to iterative search
		# -------------------------
		idx = None
		for i in xrange(l, u):
			if abs(self.__percentiles[i] - percentile) < tol:
				idx = i
				break

		# retrieve dose from index, interpolate if needed
		# -----------------------------------------------
		if idx is None: idx = u
		if abs(self.__percentiles[idx] - percentile) <= abstol:
			# return dose if available percentile bucket is close enough
			return self.__doses[idx]
		else:
			# interpolate dose by interpolating percentiles if not close enough
			alpha = self.__interpolate_percentile(self.__percentiles[i],
				self.__percentiles[i + 1], percentile)
			return alpha * self.__doses[i] + (1 - alpha) * self.__doses[i + 1]
	
	def percentile_at_dose(self, dose):
		if self.__percentiles is None: return nan

		if dose <= self.min_dose: return 100
		if dose >= self.max_dose: return 0

		# bisection retrieval of index @ percentile
		# ----------------------------------------
		u = len(self.__doses) - 1
		l = 1
		i = int(l + (u - l) / 2)

		# set tolerance based on bucket width
		tol = (self.__doses[-2] - self.__doses[-1]) / 2

		# get to within 0.5 of a dose if possible
		abstol = 0.5

		while (u - l > 5):
			# dose sorted descending
			if self.__doses[i] > dose:
				l = i
			else:
				u = i
			i = int(l + (u - l) / 2)

		# break to iterative search
		# -------------------------
		idx = None
		for i in xrange(l, u):
			if abs(self.__doses[i] - dose) < tol:
				idx = i
				break

		# retrieve percentile from index, interpolate if needed
		# -----------------------------------------------
		if idx is None: idx = u
		if abs(self.__doses[idx] - dose) <= abstol:
			# return percentile if available dose bucket is close enough
			return self.__percentiles[idx]
		else:
			# interpolate percentile by interpolating doses if not close enough
			alpha = self.__interpolate_percentile(self.__doses[i],
				self.__doses[i + 1], dose)
			return alpha * self.__percentiles[i] + (1 - alpha) * self.__percentiles[i + 1]

	@property
	def min_dose(self):
		""" Smallest dose value in DVH curve. """
		if self.__doses is None: return nan
		return self.__dose_buffer[0]

	@property
	def max_dose(self):
		""" Largest dose value in DVH curve. """
		if self.__doses is None: return nan
		return self.__dose_buffer[-1]

	@property
	def plotting_data(self):
		""" Dictionary of :mod:`matplotlib`-compatible plotting data. """
		return {'percentile' : self.__percentiles, 'dose' : self.__doses}
