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

import unittest
from numpy import nan
from numpy.linalg import norm
from numpy.random import rand

class ConradTestCase(unittest.TestCase):
	"""
	Base test class for :mod:`conrad` unit testing, extends :class:`unittest.TestCase`.
	"""

	def assert_vector_equal(self, first, second, atol=1e-7, rtol=1e-7):
		"""
		Assert ``first`` and ``second`` equal, entrywise, within tolerance.
		"""
		atol *= len(first)**0.5
		self.assertTrue( norm(first - second) <= atol + rtol * norm(second) )

	def assert_vector_notequal(self, first, second, atol=1e-7, rtol=1e-7):
		"""
		Assert ``first`` and ``second`` not equal, entrywise, within tolerance.
		"""
		first = vec(first)
		second = vec(second)
		atol *= len(first)**0.5
		self.assertTrue( norm(first - second) > atol + rtol * norm(second) )

	def assert_scalar_equal(self, first, second, atol=1e-7, rtol=1e-7):
		""" Assert ``first`` and ``second`` equal, within tolerance. """
		self.assertTrue( abs(first - second) <= atol + rtol * abs(second) )

	def assert_scalar_notequal(self, first, second, atol=1e-7, rtol=1e-7):
		""" Assert ``first`` and ``second`` not equal, within tolerance. """
		self.assertTrue( abs(first - second) > atol + rtol * abs(second) )

	def assert_nan(self, value):
		""" Assert ``value`` is :attr:``numpy.nan``. """
		condition = value is nan
		condition |= str(value) == 'nan'
		self.assertTrue( condition )

	def assert_exception(self, call=None, args=None):
		""" Assert ``call`` triggers exception, when supplied ``args``. """
		try:
			call(*args)
			self.assertTrue( False )
		except:
			self.assertTrue( True )

	def assert_no_exception(self, call=None, args=None):
		""" Assert ``call`` is exception-free, when supplied ``args``. """
		if call is None or args is None:
			return

		call(*args)
		self.assertTrue( True )

	def assert_property_exception(self, obj=None, property_name=None):
		""" Assert ``obj``.``property_name`` triggers exception. """
		try:
			val = obj.__dict__[str(property_name)]
			self.assertTrue( False )
		except:
			self.assertTrue( True )

