import unittest
from numpy import nan
from numpy.linalg import norm
from numpy.random import rand
from conrad.compat import *

class ConradTestCase(unittest.TestCase):
	def assert_vector_equal(self, first, second, atol=1e-7, rtol=1e-7):
		atol *= len(first)**0.5
		self.assertTrue( norm(first - second) <= atol + rtol * norm(second) )

	def assert_vector_notequal(self, first, second, atol=1e-7, rtol=1e-7):
		atol *= len(first)**0.5
		self.assertTrue( norm(first - second) > atol + rtol * norm(second) )

	def assert_scalar_equal(self, first, second, atol=1e-7, rtol=1e-7):
		self.assertTrue( abs(first - second) <= atol + rtol * abs(second) )

	def assert_scalar_notequal(self, first, second, atol=1e-7, rtol=1e-7):
		self.assertTrue( abs(first - second) > atol + rtol * abs(second) )

	def assert_nan(self, value):
		condition = value is nan
		condition |= str(value) == 'nan'
		self.assertTrue( condition )

	def assert_exception(self, call=None, args=None):
		try:
			call(*args)
			self.assertTrue( False )
		except:
			self.assertTrue( True )

	def assert_no_exception(self, call=None, args=None):
		if call is None or args is None:
			return

		call(*args)
		self.assertTrue( True )

	def assert_property_exception(self, obj=None, property_name=None):
		try:
			val = obj.__dict__[str(property_name)]
			self.assertTrue( False )
		except:
			self.assertTrue( True )
