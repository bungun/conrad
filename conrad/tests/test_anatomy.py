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
from conrad.medicine.structure import Structure
from conrad.medicine.anatomy import *
from conrad.tests.base import *

class AnatomyTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.A0 = A0 = rand(500, 50)
		self.A1 = A1 = rand(200, 50)
		self.structures = [
				Structure(0, 'oar', False, A=A0),
				Structure(1, 'ptv', True, A=A1)
		]

	def setUp(self):
		self.x_random = rand(50)

	def test_anatomy_init(self):
		a = Anatomy()
		self.assertTrue( isinstance(a.structures, dict) )
		self.assertTrue( len(a.structures) == 0 )
		self.assertTrue( a.is_empty )
		self.assertTrue( a.n_structures == 0 )
		self.assertTrue( a.size == 0 )
		self.assertFalse( a.plannable )

	def test_structure_add_remove(self):
		a = Anatomy()

		# add OAR
		a += Structure('label1', 'oar', False)
		self.assertFalse( a.is_empty )
		self.assertTrue( a.n_structures == 1 )
		self.assertFalse( a.plannable )

		self.assertTrue( a.size is nan )
		a['label1'].A_full = rand(500, 100)
		self.assertTrue( a.size == 500 )
		self.assertFalse( a.plannable )

		a += Structure('label2', 'ptv', True)
		self.assertFalse( a.plannable )
		a['label2'].A_full = rand(500, 100)
		self.assertTrue( a.n_structures == 2 )
		self.assertTrue( a.size == 1000 )
		self.assertTrue( a.plannable )

		self.assertTrue( 'label1' in a.labels )
		self.assertTrue( 'label2' in a.labels )

		a -= 'label1'
		self.assertTrue( a.n_structures == 1 )
		self.assertTrue( a.size == 500 )
		self.assertFalse( 'label1' in a.labels )
		self.assertTrue( a.plannable )

		a -= 'ptv'
		self.assertTrue( a.is_empty )
		self.assertTrue( a.n_structures == 0 )
		self.assertTrue( a.size == 0 )
		self.assertFalse( 'label2' in a.labels )
		self.assertFalse( a.plannable )

		structure_list = [
				Structure(0, 'oar', False),
				Structure(1, 'ptv', True)
		]

		a.structures = self.structures
		self.assertTrue( a.n_structures == 2 )
		self.assertTrue( 0 in a.labels )
		self.assertTrue( 1 in a.labels )

		a2 = Anatomy(self.structures)
		self.assertTrue( a2.n_structures == 2 )
		self.assertTrue( 0 in a2.labels )
		self.assertTrue( 1 in a2.labels )

		a3 = Anatomy(a)
		self.assertTrue( a3.n_structures == 2 )
		self.assertTrue( 0 in a3.labels )
		self.assertTrue( 1 in a3.labels )

	def test_dose_calc_and_summary(self):
		a = Anatomy(self.structures)
		y0 = self.A0.dot(self.x_random)
		y1 = self.A1.dot(self.x_random)

		a.calculate_doses(self.x_random)
		self.assert_vector_equal( y0, a[0].y )
		self.assert_vector_equal( y1, a[1].y )

		ds = a.dose_summary_data()
		for s in self.structures:
			self.assertTrue( s.label in ds )
			self.assertTrue( 'min' in ds[s.label] )
			self.assertTrue( 'mean' in ds[s.label] )
			self.assertTrue( 'max' in ds[s.label] )
			self.assertTrue( 'D2' in ds[s.label] )
			self.assertTrue( 'D98' in ds[s.label] )

		ds = a.dose_summary_data(percentiles=[1])
		for s in self.structures:
			self.assertTrue( s.label in ds )
			self.assertTrue( 'min' in ds[s.label] )
			self.assertTrue( 'mean' in ds[s.label] )
			self.assertTrue( 'max' in ds[s.label] )
			self.assertTrue( 'D1' in ds[s.label] )
			self.assertFalse( 'D2' in ds[s.label] )
			self.assertFalse( 'D98' in ds[s.label] )

		ds = a.dose_summary_data(percentiles=[10, 22, 83])
		for s in self.structures:
			self.assertTrue( s.label in ds )
			self.assertTrue( 'D10' in ds[s.label] )
			self.assertTrue( 'D22' in ds[s.label] )
			self.assertTrue( 'D83' in ds[s.label] )


		ds = a.dose_summary_data(percentiles=xrange(10, 100, 10))
		for s in self.structures:
			self.assertTrue( s.label in ds )
			for p in xrange(10, 100, 10):
				self.assertTrue( 'D{}'.format(p) in ds[s.label] )

		ds = a.dose_summary_string
		for s in self.structures:
			self.assertTrue( s.summary_string in ds )