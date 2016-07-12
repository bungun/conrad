import unittest
from conrad.physics.units import *

class TestUnits(unittest.TestCase):
	def test_units(self):
		a = 32 * CM()
		self.assertTrue( isinstance(a, Length) )
		self.assertTrue( isinstance(a, CM) )
		self.assertTrue( isinstance(a.to_cm(), CM) )
		self.assertEqual(str(a), '32 cm')

		b= 10 * MM()
		self.assertTrue( isinstance(b, Length) )
		self.assertTrue( isinstance(b, MM) )
		self.assertTrue( isinstance(b.to_cm(), CM) )
		self.assertEqual(str(b), '10 mm')

		self.assertTrue( isinstance(a * b, Area) )
		self.assertTrue( isinstance(a * b, CM2) )
		self.assertEqual(str(a * b), '32.0 cm^2')
		self.assertTrue( isinstance(b * a, Area) )
		self.assertTrue( isinstance(b * a, MM2) )
		self.assertEqual(str(b * a), '3200.0 mm^2')

		c = 2 * MM()
		self.assertTrue( isinstance(a * b * c, Volume) )
		self.assertTrue( isinstance(a * b * c, CM3) )
		self.assertTrue( isinstance(a * (b * c), CM3) )
		self.assertTrue( isinstance((a * b) * c, CM3) )
		self.assertTrue( isinstance(b * a * c, MM3) )
		self.assertEqual(str(a * b * c), str((b * a * c).to_cm3()))