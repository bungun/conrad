"""
Module defines base class for :mod:`conrad` unit testing.
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

import numpy as np
import numpy.linalg as la
import unittest

class ConradTestCase(unittest.TestCase):
	"""
	Base test class for :mod:`conrad` unit testing, extends :class:`unittest.TestCase`.
	"""

	def assert_vector_equal(self, first, second, atol=1e-7, rtol=1e-7):
		"""
		Assert ``first`` and ``second`` equal, entrywise, within tolerance.
		"""
		atol *= len(first)**0.5
		self.assertLessEqual(
				la.norm(first - second), atol + rtol * la.norm(second) )

	def assert_vector_notequal(self, first, second, atol=1e-7, rtol=1e-7):
		"""
		Assert ``first`` and ``second`` not equal, entrywise, within tolerance.
		"""
		first = vec(first)
		second = vec(second)
		atol *= len(first)**0.5
		self.assertGreater(
				la.norm(first - second), atol + rtol * la.norm(second) )

	def assert_scalar_equal(self, first, second, atol=1e-7, rtol=1e-7):
		""" Assert ``first`` and ``second`` equal, within tolerance. """
		self.assertLessEqual( abs(first - second), atol + rtol * abs(second) )

	def assert_scalar_notequal(self, first, second, atol=1e-7, rtol=1e-7):
		""" Assert ``first`` and ``second`` not equal, within tolerance. """
		self.assertGreater( abs(first - second), atol + rtol * abs(second) )

	def assert_nan(self, value):
		""" Assert ``value`` is :attr:``numpy.np.nan``. """
		self.assertTrue( value is np.nan or str(value) == 'nan' )

	def assert_not_nan(self, value):
		""" Assert ``value`` is not :attr:``numpy.np.nan``. """
		self.assertTrue( value is not np.nan and str(value) != 'nan' )

