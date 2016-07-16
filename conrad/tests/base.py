import unittest
from numpy.linalg import norm

class ConradTestCase(unittest.TestCase):
	def assert_vector_equal(self, first, second, atol, rtol):
		atol *= len(first)**0.5
		self.assertTrue( norm(first - second) <= atol + rtol * norm(second) )

	def assert_vector_notequal(self, first, second, atol, rtol):
		atol *= len(first)**0.5
		self.assertTrue( norm(first - second) > atol + rtol * norm(second) )

	def assert_scalar_equal(self, first, second, atol, rtol):
		self.assertTrue( abs(first - second) <= atol + rtol * abs(second) )

	def assert_scalar_notequal(self, first, second, atol, rtol):
		self.assertTrue( abs(first - second) > atol + rtol * abs(second) )
