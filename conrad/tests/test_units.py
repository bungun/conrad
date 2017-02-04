"""
Unit tests for :mod:`conrad.physics.units`.
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
from conrad.physics.units import *
from conrad.tests.base import *

class TestUnits(ConradTestCase):
	def test_percent(self):
		p = Percent()
		p = 10 * Percent()
		self.assertIsInstance( p, Percent )
		self.assertEqual( p.value, 10 )
		self.assertEqual( p.fraction, 0.1)
		self.assertEqual( p, 10 * Percent() )
		self.assertEqual( str(p), '10.0%' )

	def test_length_units(self):
		a = 32 * CM()
		self.assertIsInstance( a, Length )
		self.assertIsInstance( a, CM )
		self.assert_scalar_equal( 32, a.value, 1e-7, 1e-7)
		self.assertIsInstance( a.to_mm, MM )
		self.assert_scalar_equal( 320, a.to_mm.value, 1e-7, 1e-7)
		self.assertEqual( str(a), '32.0 cm' )

		b = 10 * MM()
		self.assertIsInstance( b, Length )
		self.assertIsInstance( b, MM )
		self.assert_scalar_equal( 10, b.value, 1e-7, 1e-7)
		self.assertIsInstance( b.to_cm, CM )
		self.assert_scalar_equal( 1, b.to_cm.value, 1e-7, 1e-7)
		self.assertEqual( str(b), '10.0 mm' )

		self.assertIsInstance( a * b, Area )
		self.assertIsInstance( a * b, CM2 )
		self.assert_scalar_equal( 32, (a * b).value, 1e-7, 1e-7)
		self.assertEqual( str(a * b), '32.0 cm^2' )
		self.assertIsInstance( b * a, Area )
		self.assertIsInstance( b * a, MM2 )
		self.assert_scalar_equal( 3200, (b * a).value, 1e-7, 1e-7)
		self.assertEqual( str(b * a), '3200.0 mm^2' )

		c = 2 * MM()
		self.assertIsInstance( a * b * c, Volume )
		self.assertIsInstance( a * b * c, CM3 )
		self.assertIsInstance( a * (b * c), CM3 )
		self.assertIsInstance( (a * b) * c, CM3 )
		self.assertIsInstance( b * a * c, MM3 )
		self.assert_scalar_equal( (a * b * c).value, (b * a * c).to_cm3.value,
								  1e-7, 1e-7 )
		self.assertEqual( str(a * b * c), str((b * a * c).to_cm3 ) )

		self.assertEqual( 3 * CM(), 3e1 * MM() )
		self.assertEqual( 3 * CM2(), 3e2 * MM2() )
		self.assertEqual( 3 * CM3(), 3e3 * MM3() )

		self.assertEqual( 3 * MM(), 3e-1 * CM() )
		self.assertEqual( 3 * MM2(), 3e-2 * CM2() )
		self.assertEqual( 3 * MM3(), 3e-3 * CM3() )

	def test_dose_units(self):
		d = 10 * Gy
		self.assertIsInstance( d, DeliveredDose )
		self.assertIsInstance( d, Gray )
		self.assert_scalar_equal( 10, d.value, 1e-7, 1e-7)
		self.assertIsInstance( d.to_cGy, centiGray )
		self.assert_scalar_equal( 1000, d.to_cGy.value, 1e-7, 1e-7)
		self.assertEqual( str(d), '10.0 Gy' )

		e = 2000 * cGy
		self.assertIsInstance( e, DeliveredDose )
		self.assertIsInstance( e, centiGray )
		self.assert_scalar_equal( 2000, e.value, 1e-7, 1e-7)
		self.assertIsInstance( e.to_Gy, Gray )
		self.assert_scalar_equal( 20, e.to_Gy.value, 1e-7, 1e-7)
		self.assertEqual( str(e), '2000.0 cGy' )

		self.assertEqual( 10 * Gy, 1000 * cGy )
