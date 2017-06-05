"""
Defines the :class:`DVH` (dose volume histogram) object for
converting structure dose vectors to plottable DVH data sets.
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
from conrad.medicine.dose.constraints import *

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
		if n_voxels is None or n_voxels is np.nan or n_voxels < 1:
			raise ValueError('argument "n_voxels" must be an integer > 0')

		self.__dose_buffer = np.zeros(int(n_voxels))
		self.__stride = 1 * (n_voxels < maxlength) + int(n_voxels / maxlength)
		length = len(self.__dose_buffer[::self.__stride]) + 1
		self.__doses = np.zeros(length)
		self.__percentiles = np.zeros(length)
		self.__percentiles[0] = 100.
		self.__percentiles[1:] = np.linspace(100, 0, length - 1)
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
							 '[{},{}]\ntarget value: {}'.format(p1, p2, p_des))
		else:
			# alpha * p1 + (1 - alpha) * p2 = p_des
			# (p1 - p2) * alpha = p_des - p2
			# alpha = (p_des - p2) / (p1 - p2)
			return float(p_des - p2) / float(p1 - p2)

	def percentile_at_dose(self, dose):
		"""
		Read off DVH curve to get precentile value at ``dose``.

		Arguments:
			dose (:obj:`int`, :obj:`float`, or :class:`DeliveredDose`):
				Quertied dose for which to retrieve the corresponding
				percentile. Assumed to have same units as DVH data.

		Returns:
			Percentile value from DVH curve corresponding to queried
			dose, or :attr:`~numpy.np.nan` if the curve has not been
			populated with data.
		"""
		if isinstance(dose, DeliveredDose):
			dose = dose.value

		if self.__doses is None: return np.nan

		return 100. * (
				sum(self.__dose_buffer < dose) /
				float(self.__dose_buffer.size))

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
			percentile (:obj:`int`, :obj:`float` or :class:`Percent`):
				Queried percentile for which to retrieve corresponding
				dose level.

		Returns:
			Dose value from DVH curve corresponding to queried
			percentile, or :attr:`~numpy.np.nan` if the curve has not been
			populated with data.
		"""
		if isinstance(percentile, Percent):
			percentile = percentile.value

		if self.__doses is None: return np.nan

		if percentile == 100:
			return self.min_dose
		if percentile == 0:
			return self.max_dose

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
		""" Smallest dose value in DVH curve. """
		if self.__doses is None: return np.nan
		return self.__dose_buffer[0]

	@property
	def max_dose(self):
		""" Largest dose value in DVH curve. """
		if self.__doses is None: return np.nan
		return self.__dose_buffer[-1]

	@property
	def plotting_data(self):
		""" Dictionary of :mod:`matplotlib`-compatible plotting data. """
		return {'percentile' : self.__percentiles, 'dose' : self.__doses.copy()}

	def resample(self, maxlength):
		"""
		Re-sampled copy of this :class`DVH`

		Args:
			maxlength (:obj:`int`): Maximum length at which to series
				re-sample data.

		Returns:
			:class:`DVH`: Re-sampled DVH curve; return original curve
			if ``maxlength`` is ``None``.

		"""
		if maxlength is None:
			return self

		dvh = DVH(self.__dose_buffer.size, maxlength=int(maxlength))
		dvh.data = self.__dose_buffer
		return dvh