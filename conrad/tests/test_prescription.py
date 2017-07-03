"""
Unit tests for :mod:`conrad.medicine.prescription`.
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

import os

from conrad.medicine.dose import D
from conrad.medicine.prescription import *
from conrad.tests.base import *

class PrescriptionTestCase(ConradTestCase):
	def setUp(self):
		# structure labels
		self.LABEL1 = 1
		self.LABEL2 = 4
		self.LABEL3 = 7

		# structure names
		self.NAME1 = 'PTV'
		self.NAME1_VARIANTS = ('PTV', 'PTV_JSON', 'PTV_YAML')
		self.NAME2 = 'OAR_RESILIENT'
		self.NAME3 = 'OAR_SENSITIVE'

		# structure doses
		self.DOSE1 = 1 * Gy
		self.DOSE2 = self.DOSE3 = 0 * Gy

		# constraints
		self.c11 = D(85) > 0.8 * self.DOSE1
		self.c12 = D(1) < 1.2 * Gy
		self.c21 = D(50) < 0.6 * Gy
		self.c22 = D(2) < 0.8 * Gy
		self.c31 = D(50) < 0.3 * Gy
		self.c32 = D(2) < 0.6 * Gy

	def validate_prescription_contents(self, prescription):
		""" check that input "prescription" matches the following content:

			- name : PTV_JSON
			  label : 1
			  is_target : Yes
			  dose : 1 Gy
			  constraints :
			  - D85 > 0.8 rx
			  - D1 < 1.2 Gy

			- name : OAR_RESILIENT
			  label : 4
			  is_target : No
			  dose :
			  constraints :
			  - D50 < 0.6 Gy
			  - D2 < 0.8 Gy

			- name : OAR_SENSITIVE
			  label : 7
			  is_target : No
			  dose :
			  constraints :
			  - D50 < 0.2 Gy
			  - D2 < 0.6 Gy

			  these data are mirrored in:
			  	- the setUp() method of this PrescriptionTestCase object
			  	- the local file yaml_rx.yml
			  	- the local file json_rx.json

		"""
		self.assertIsInstance( prescription, Prescription )
		rx = prescription

		# structure labels
		LABEL1 = self.LABEL1
		LABEL2 = self.LABEL2
		LABEL3 = self.LABEL3

		self.assertIn( LABEL1, rx.structure_dict)
		self.assertIn( LABEL2, rx.structure_dict)
		self.assertIn( LABEL3, rx.structure_dict)
		self.assertIsInstance( rx.structure_dict[LABEL1], Structure )
		self.assertIsInstance( rx.structure_dict[LABEL2], Structure )
		self.assertIsInstance( rx.structure_dict[LABEL3], Structure )
		self.assertIn( rx.structure_dict[LABEL1].name, self.NAME1_VARIANTS )
		self.assertEqual( rx.structure_dict[LABEL2].name, self.NAME2 )
		self.assertEqual( rx.structure_dict[LABEL2].name, self.NAME2 )
		self.assertEqual( rx.structure_dict[LABEL1].dose, self.DOSE1 )
		self.assertEqual( rx.structure_dict[LABEL2].dose, self.DOSE2 )
		self.assertEqual( rx.structure_dict[LABEL3].dose, self.DOSE3 )

		self.assertIn( LABEL1, rx.constraint_dict)
		self.assertIn( LABEL2, rx.constraint_dict)
		self.assertIn( LABEL3, rx.constraint_dict)
		self.assertIn( str(self.c11), str(rx.constraint_dict[LABEL1]) )
		self.assertIn( str(self.c12), str(rx.constraint_dict[LABEL1]) )
		self.assertIn( str(self.c21), str(rx.constraint_dict[LABEL2]) )
		self.assertIn( str(self.c22), str(rx.constraint_dict[LABEL2]) )
		self.assertIn( str(self.c31), str(rx.constraint_dict[LABEL3]) )
		self.assertIn( str(self.c32), str(rx.constraint_dict[LABEL3]) )

	def test_prescription_init(self):
		rx = Prescription()
		self.assertIsInstance( rx.constraint_dict, dict )
		self.assertEqual( len(rx.constraint_dict), 0 )

		self.assertIsInstance( rx.structure_dict, dict )
		self.assertEqual( len(rx.structure_dict), 0 )

		self.assertIsInstance( rx.rx_list, list )
		self.assertEqual( len(rx.rx_list), 0 )

		slist = [
				Structure(self.LABEL1, self.NAME1, True, dose=self.DOSE1),
				Structure(self.LABEL2, self.NAME2, False),
				Structure(self.LABEL3, self.NAME3, False)
		]
		for s in slist:
			rx.add_structure_to_dictionaries(s)
			self.assertIn( s.label, rx.structure_dict )
			self.assertIn( s.label, rx.constraint_dict )
			self.assertEqual( rx.constraint_dict[s.label].size, 0 )

		rx.constraint_dict[self.LABEL1] += self.c11
		rx.constraint_dict[self.LABEL1] += self.c12
		rx.constraint_dict[self.LABEL2] += self.c21
		rx.constraint_dict[self.LABEL2] += self.c22
		rx.constraint_dict[self.LABEL3] += self.c31
		rx.constraint_dict[self.LABEL3] += self.c32
		self.validate_prescription_contents(rx)

		# test copy constructor
		rx2 = Prescription(rx)
		self.validate_prescription_contents(rx2)

	def test_prescription_digest_python(self):
		rx_data = [{
			'name' : 'PTV',
			'label' : 1,
			'is_target' : True,
			'dose' : '1 Gy',
			'constraints' : ['D85 > 0.8 rx', 'D1 < 1.2 Gy']
		}, {
			'name' : 'OAR_RESILIENT',
			'label' : 4,
			'is_target' : False,
			'dose' : None,
			'constraints' : ['D50 < 0.6 Gy', 'D2 < 0.8 Gy']
		}, {
			'name' : 'OAR_SENSITIVE',
			'label' : 7,
			'is_target' : False,
			'dose' : None,
			'constraints' : ['D50 < 0.3 Gy', 'D2 < 0.6 Gy']
		}]

		rx = Prescription(rx_data)
		self.validate_prescription_contents(rx)

	def test_prescription_digest_yaml(self):
		f = os.path.join(
				os.path.abspath(os.path.dirname(__file__)), 'yaml_rx.yml')
		rx = Prescription(f)
		self.validate_prescription_contents(rx)

	def test_prescription_digest_json(self):
		f = os.path.join(
				os.path.abspath(os.path.dirname(__file__)), 'json_rx.json')
		rx = Prescription(f)
		self.validate_prescription_contents(rx)

	def test_prescription_report(self):
		pass

	def test_prescription_report_string(self):
		pass

