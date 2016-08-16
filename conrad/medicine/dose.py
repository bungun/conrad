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
from time import clock
from hashlib import sha1
from numpy import zeros, linspace, nan, ceil

from conrad.compat import *
from conrad.physics.units import DeliveredDose, cGy, Gy, Percent, Volume, cm3

class __ConstraintRelops(object):
	GEQ = '>='
	LEQ = '<='
	INDEFINITE = '<>'

RELOPS = __ConstraintRelops()

class Constraint(object):
	def __init__(self):
		self.__dose = nan
		self.__threshold = None
		self.__relop = RELOPS.INDEFINITE
		self.__slack = 0.
		self.__priority = 1
		self.__prescription_dose = nan * Gy
		self.__dual = nan

	@property
	def resolved(self):
		if isinstance(self.dose, (DeliveredDose, Percent)):
			dose_resolved = self.dose.value is not nan
		else:
			dose_resolved = False
		relop_resolved = self.relop != RELOPS.INDEFINITE
		threshold_resolved = self.threshold is not None
		return dose_resolved and relop_resolved and threshold_resolved

	@property
	def threshold(self):
		return self.__threshold

	@threshold.setter
	def threshold(self, threshold):
		self.__threshold = threshold

	@property
	def relop(self):
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
		if self.__relop == RELOPS.INDEFINITE:
			raise ValueError('{} object field "relop" not initialized, '
							 'identity as upper/lower constraint undefined'
							 ''.format(Constraint))
		return self.__relop == RELOPS.LEQ

	@property
	def dose(self):
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
		return self.__dual > 0.

	@property
	def dual_value(self):
		return self.__dual

	@dual_value.setter
	def dual_value(self, dual_value):
		self.__dual = float(dual_value)

	@property
	def slack(self):
	    return self.__slack

	@slack.setter
	def slack(self, slack):
		if isinstance(slack, (int, float)):
			if slack < 0:
				raise ValueError('argument "slack" must be nonnegative by '
								 'convention')
			self.__slack = max(0., float(slack))
		else:
			raise TypeError('argument "slack" must be of type {} with value '
					  		'>= 0'.format(float))

	@property
	def dose_achieved(self):
		sign = +1 if self.upper else -1
		return self.dose + sign * self.__slack

	@property
	def priority(self):
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
		return self.__relop.replace('=', '')

	def __lt__(self, other):
		self.relop = RELOPS.LEQ
		self.dose = other
		return self

	def __le__(self, other):
		return self.__lt__(other)

	def __gt__(self, other):
		self.relop = RELOPS.GEQ
		self.dose = other
		return self

	def __ge__(self, other):
		return self.__gt__(other)

	def __eq__(self, other):
		if not isinstance(other, Constraint):
			raise TypeError('equality comparison for {} object only defined '
							'when both operands of type {}'.format(Constraint,
							Constraint))

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
		thresh = self.__threshold
		thresh = str(thresh.value) if isinstance(thresh, Percent) else thresh
		return str('D{} {} {}'.format(thresh, self.__relop, self.__dose))

class PercentileConstraint(Constraint):
	def __init__(self, percentile=None, relop=None, dose=None):
		Constraint.__init__(self)
		if relop is not None:
			self.relop = relop
		if dose is not None:
			self.dose = dose
		if percentile is not None:
			self.percentile = percentile

	@property
	def percentile(self):
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
		return {
				'type': 'percentile',
				'percentile' : 2 * [self.percentile.value],
				'dose' :[self.dose.value, self.dose_achieved.value],
				'symbol' : self.symbol
				}

	def get_maxmargin_fulfillers(self, y, had_slack=False):
		"""
		given dose vector y, get the indices of the voxels that
		fulfill this dose constraint (self) with maximum margin

		given len(y), if m voxels are required to respect the
		dose constraint exactly, y is assumed to contain
		at least m entries that respect the constraint
		(for instance, y is generated by a convex program
		that includes a convex restriction of the dose constraint)


		procedure:
		- get margins: (y - self.dose_requested)
		- sort margin indices by margin values
		- if upper bound, return indices of p most negative entries
			(first p of sorted indices; numpy.sort sorts small to large)
		- if lower bound, return indices p most positive entries
			(last p of sorted indices; numpy.sort sorts small to large)

		p = percent non-violating * structure size
			= percent non-violating * len(y)

		"""

		fraction = self.percentile.fraction
		non_viol = (1 - fraction) if self.upper else fraction
		n_returned = int(ceil(non_viol * len(y)))

		start = 0 if self.upper else -n_returned
		end = n_returned if self.upper else None
		dose = self.dose_achieved.value if had_slack else self.dose.value
		return (y - dose).argsort()[start:end]


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
	def __init__(self, relop=None, dose=None):
		Constraint.__init__(self)
		if relop is not None:
			self.relop = relop
		if dose is not None:
			self.dose = dose
		self.threshold = 'mean'

	@property
	def plotting_data(self):
		return {'type': 'mean',
			'dose' : [self.dose.value, self.dose_achieved.value],
			'symbol' : self.symbol}

class MinConstraint(Constraint):
	def __init__(self, relop=None, dose=None):
		Constraint.__init__(self)
		if relop is not None:
			self.relop = relop
		if dose is not None:
			self.dose = dose
		self.threshold = 'min'

	@property
	def plotting_data(self):
		return {'type': 'min',
			'dose' : [self.dose.value, self.dose_achieved.value],
			'symbol' : self.symbol}

class MaxConstraint(Constraint):
	def __init__(self, relop=None, dose=None):
		Constraint.__init__(self)
		if relop is not None:
			self.relop = relop
		if dose is not None:
			self.dose = dose
		self.threshold = 'max'

	@property
	def plotting_data(self):
		return {'type': 'max',
			'dose' :[self.dose.value, self.dose_achieved.value],
			'symbol' : self.symbol}

def D(threshold, relop=None, dose=None):
	if isinstance(threshold, str):
		if threshold in ('mean', 'Mean'):
			return MeanConstraint(relop=relop, dose=dose)
		elif threshold in ('min', 'Min', 'minimum', 'Minimum') or threshold == 100:
			return MinConstraint(relop=relop, dose=dose)
		elif threshold in ('max', 'Max', 'maximum', 'Maximum') or threshold == 0:
			return MaxConstraint(relop=relop, dose=dose)
	elif isinstance(threshold, (int, float, Percent)):
		return PercentileConstraint(percentile=threshold, relop=relop, dose=dose)
	else:
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
	def __init__(self):
		self.items = {}
		self.last_key = None

	@staticmethod
	def __keygen(constraint):
		return sha1(str(
				str(clock()) +
				str(constraint.dose) +
				str(constraint.threshold) +
				str(constraint.relop)
			).encode('utf-8')).hexdigest()[:7]

	def __getitem__(self, key):
		return self.items[key]

	def __iter__(self):
		return self.items.__iter__()

	def __next__(self):
		return self.items.__next__()

	def next(self):
		return self.items.next()

	def iteritems(self):
		return self.items.items()

	def itervalues(self):
		return self.items.values()

	def __iadd__(self, other):
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
		return len(self.items)

	@property
	def keys(self):
		return self.items.keys()

	@property
	def list(self):
		return self.items.values()


	@property
	def mean_only(self):
		meantest = lambda c : isinstance(c, MeanConstraint)
		if self.size == 0:
			return True
		else:
			return all(listmap(meantest, self.itervalues()))

	def contains(self, constr):
		return constr in self.items.values()

	def clear(self):
		self.items = {}

	@property
	def plotting_data(self):
		return [(key, dc.plotting_data) for key, dc in self.items.items()]

	def __str__(self):
		out = '(keys):\t (constraints)\n'
		for key, constr in self.items.items():
			out += key + ':\t' + str(constr) + '\n'
		return out

class DVH(object):
	"""
	TODO: DVHCurve docstring
	"""

	MAX_LENGTH = 1000

	def __init__(self, n_voxels, maxlength=MAX_LENGTH):
		""" TODO: docstring """
		if n_voxels is None or n_voxels is nan or n_voxels < 1:
			raise ValueError('argument "n_voxels" must be an integer > 1')

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
		return self.__DATA_ENTERED

	@property
	def data(self):
	    return self.__doses[1:]

	@data.setter
	def data(self, y):
		""" TODO: docstring """
		if len(y) != self.__dose_buffer.size:
			raise ValueError('dimension mismatch: length of argument "y" '
							 'must be {}'.format(self.__dose_buffer.size))

		self.__dose_buffer[:] = y[:]
		self.__dose_buffer.sort()
		self.__doses[1:] = self.__dose_buffer[::self.__stride]
		self.__DATA_ENTERED = True

	@staticmethod
	def __interpolate_percentile(p1, p2, p_des):
		""" TODO: docstring """
		# alpha * p1 + (1 - alpha) * p2 = p_des
		# (p1 - p2) * alpha = p_des - p2
		# alpha = (p_des - p2) / (p1 - p2)
		if p1 == p2 and p_des == p1:
			return 1.
		elif p1 == p2 and p_des != p1:
			raise ValueError('arguments "p1", "p2" must be distinct to '
							 'perform interpolation.\nprovided interval: '
							 '[{},{}]\ntaget value: {}'.format(p1, p2, p_des))
		else:
			return float(p_des - p2) / float(p1 - p2)

	def dose_at_percentile(self, percentile):
		""" TODO: docstring """
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

	@property
	def min_dose(self):
		""" TODO: docstring """
		if self.__doses is None: return nan
		return self.__dose_buffer[0]

	@property
	def max_dose(self):
		""" TODO: docstring """
		if self.__doses is None: return nan
		return self.__dose_buffer[-1]

	@property
	def plotting_data(self):
		""" TODO: docstring """
		return {'percentile' : self.__percentiles, 'dose' : self.__doses}
