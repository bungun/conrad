from conrad.compat import *
from conrad.physics.units import *
from conrad.tests.base import *

class TestUnits(ConradTestCase):
	def test_percent(self):
		p = 10 * Percent()
		self.assertTrue( isinstance(p, Percent) )
		self.assertTrue( p.value == 10 )
		self.assertTrue( p.fraction == 0.1)
		self.assertTrue( p == 10 * Percent() )
		self.assertTrue( str(p) == '10.0%' )

	def test_length_units(self):
		a = 32 * CM()
		self.assertTrue( isinstance(a, Length) )
		self.assertTrue( isinstance(a, CM) )
		self.assert_scalar_equal( 32, a.value, 1e-7, 1e-7)
		self.assertTrue( isinstance(a.to_mm, MM) )
		self.assert_scalar_equal( 320, a.to_mm.value, 1e-7, 1e-7)
		self.assertEqual( str(a), '32.0 cm' )

		b = 10 * MM()
		self.assertTrue( isinstance(b, Length) )
		self.assertTrue( isinstance(b, MM) )
		self.assert_scalar_equal( 10, b.value, 1e-7, 1e-7)
		self.assertTrue( isinstance(b.to_cm, CM) )
		self.assert_scalar_equal( 1, b.to_cm.value, 1e-7, 1e-7)
		self.assertEqual( str(b), '10.0 mm' )

		self.assertTrue( isinstance(a * b, Area) )
		self.assertTrue( isinstance(a * b, CM2) )
		self.assert_scalar_equal( 32, (a * b).value, 1e-7, 1e-7)
		self.assertEqual( str(a * b), '32.0 cm^2' )
		self.assertTrue( isinstance(b * a, Area) )
		self.assertTrue( isinstance(b * a, MM2) )
		self.assert_scalar_equal( 3200, (b * a).value, 1e-7, 1e-7)
		self.assertEqual( str(b * a), '3200.0 mm^2' )

		c = 2 * MM()
		self.assertTrue( isinstance(a * b * c, Volume) )
		self.assertTrue( isinstance(a * b * c, CM3) )
		self.assertTrue( isinstance(a * (b * c), CM3) )
		self.assertTrue( isinstance((a * b) * c, CM3) )
		self.assertTrue( isinstance(b * a * c, MM3) )
		self.assert_scalar_equal( (a * b * c).value, (b * a * c).to_cm3.value,
								  1e-7, 1e-7 )
		self.assertEqual( str(a * b * c), str((b * a * c).to_cm3 ) )

		self.assertTrue( 3 * CM() == 3e1 * MM() )
		self.assertTrue( 3 * CM2() == 3e2 * MM2() )
		self.assertTrue( 3 * CM3() == 3e3 * MM3() )

		self.assertTrue( 3 * MM() == 3e-1 * CM() )
		self.assertTrue( 3 * MM2() == 3e-2 * CM2() )
		self.assertTrue( 3 * MM3() == 3e-3 * CM3() )


	def test_dose_units(self):
		d = 10 * Gy
		self.assertTrue( isinstance(d, DeliveredDose) )
		self.assertTrue( isinstance(d, Gray) )
		self.assert_scalar_equal( 10, d.value, 1e-7, 1e-7)
		self.assertTrue( isinstance(d.to_cGy, centiGray) )
		self.assert_scalar_equal( 1000, d.to_cGy.value, 1e-7, 1e-7)
		self.assertEqual( str(d), '10.0 Gy' )

		e = 2000 * cGy
		self.assertTrue( isinstance(e, DeliveredDose) )
		self.assertTrue( isinstance(e, centiGray) )
		self.assert_scalar_equal( 2000, e.value, 1e-7, 1e-7)
		self.assertTrue( isinstance(e.to_Gy, Gray) )
		self.assert_scalar_equal( 20, e.to_Gy.value, 1e-7, 1e-7)
		self.assertEqual( str(e), '2000.0 cGy' )

		self.assertTrue( 10 * Gy == 1000 * cGy )
