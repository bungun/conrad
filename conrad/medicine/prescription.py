"""
Define :class:`Prescription` and methods for parsing prescription data
from python objects as well as JSON- or YAML-formatted files.

Parsing methods expect the following formats.

YAML::

	- name : PTV
	  label : 1
	  is_target: Yes
	  dose : 35.
	  constraints:
	  - "D90 >= 32.3Gy"
	  - "D1 <= 1.1rx"

	- name : OAR1
	  label : 2
	  is_target: No
	  dose :
	  constraints:
	  - "D95 <= 20Gy"
	  - "V30 Gy <= 20%"

Python :obj:`list` of :obj:`dict` (JSON approximately the same)::

	[{
		"name" : "PTV",
		"label" : 1,
		"is_target" : True,
		"dose" : 35.,
		"constraints" : ["D1 <= 1.1rx", "D90 >= 32.3Gy"]
	}, {
 		"name" : "OAR1",
	  	"label" : 2,
	  	"is_target" : False,
	  	"dose" : None,
	  	"constraints" : ["D95 <= 20Gy"]
	}]

JSON verus Python syntax differences:
	- ``true``/``false`` instead of ``True``/``False``
	- ``null`` instead of ``None``

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
import json, yaml
import traceback

from conrad.physics.units import Gy
from conrad.physics.string import dose_from_string
from conrad.medicine.structure import Structure
from conrad.medicine.anatomy import Anatomy
from conrad.medicine.dose import eval_constraint, ConstraintList

class Prescription(object):
	"""
	Class for specifying structures with dose targets and constraints.

	Attributes:
		constraint_dict (:obj:`dict`): Dictionary of
			:class:`ConstraintList` objects, keyed by structure labels.
		structure_dict (:obj:`dict`): Diciontionary of
			:class:`Structure` objects, keyed by structure labels.
		rx_list (:obj:`list`): List of dictionaries representation of
			prescription.
	"""

	def __init__(self, prescription_data=None):
		"""
		Intialize empty or populated :class:`Prescription` instance.

		Arguments:
			prescription_data (optional): Data to parse as prescription.
				If input is of type :class:`Prescription`, intializer
				acts as a copy constructor.
		"""
		self.constraint_dict = {}
		self.structure_dict = {}
		self.rx_list = []
		if isinstance(prescription_data, Prescription):
			self.constraint_dict = prescription_data.constraint_dict
			self.structure_dict = prescription_data.structure_dict
			self.rx_list = prescription_data.rx_list
		elif prescription_data:
			self.digest(prescription_data)

	def add_structure_to_dictionaries(self, structure):
		"""
		Add a new structure to internal representation of prescription.

		Arguments:
			structure (:class:`Structure`): Structure added to
				:attr:`Prescription.structure_dict`. An corresponding,
				empty constraint list is added to
				:attr:`Prescription.constraint_dict`.

		Returns:
			None

		Raises:
			TypeError: If ``structure`` not a :class:`Structure`.
		"""
		if not isinstance(structure, Structure):
			raise TypeError('argumet "Structure" must be of type {}'
							''.format(Structure))
		self.structure_dict[structure.label] = structure
		self.constraint_dict[structure.label] = ConstraintList()

	def digest(self, prescription_data):
		"""
		Populate :class:`Prescription`'s structures and dose constraints.

		Specifically, for each entry in ``prescription_data``, construct
		a :class:`Structure` to capture structure data (e.g., name,
		label), as well as a corresponding but separate
		:class:`ConstraintList` object to capture any dose constraints
		specified for the structure.

		Add each such structure to :attr:`Prescription.structure_dict`,
		and each such constraint list to
		:attr:`Prescription.constraint_dict`. Build or copy a "list of
		dictionaries" representation of the prescription data, assign to
		:attr:`Prescription.rx_list`.

		Arguments:
			prescription_data: Input to be parsed for structure and dose
				constraint data. Accepted formats include :obj:`str`
				specifying a valid path to a suitably-formatted JSON or
				YAML file, or a suitably-formatted :obj:`list` of
				:obj:`dict` objects.

		Returns:
			None

		Raises:
			TypeError: If input not of type :obj:`list` or a :obj:`str`
				specfying a valid path to file that can be loaded with
				the :meth:`json.load` or :meth:`yaml.safe_load` methods.
		"""
		err = None
		data_valid = False
		rx_list = []

		# read prescription data from list
		if isinstance(prescription_data, list):
			rx_list = prescription_data
			data_valid = True

		# read presription data from file
		if isinstance(prescription_data, str):
			if os.path.exists(prescription_data):
				try:
					f = open(prescription_data)
					if '.json' in prescription_data:
						rx_list = json.load(f)
					else:
						rx_list = yaml.safe_load(f)
					f.close
					data_valid = True
				except:
					err = traceback.format_exc()

		if not data_valid:
			if err is not None:
				print(err)
			raise TypeError(
					'input prescription_data expected to be a list or '
					'the path to a valid JSON or YAML file.')

		try:
			for item in rx_list:
				rx_dose = None
				label = item['label']
				name = item['name']
				dose = 0 * Gy
				is_target = bool(item['is_target'])
				if is_target:
					if isinstance(item['dose'], (float, int)):
						rx_dose = dose = float(item['dose']) * Gy
					else:
						rx_dose = dose = dose_from_string(item['dose'])

				s = Structure(label, name, is_target, dose=dose)
				self.add_structure_to_dictionaries(s)

				if 'constraints' in item:
					if item['constraints'] is not None:
						for string in item['constraints']:
							self.constraint_dict[label] += eval_constraint(
									string, rx_dose=rx_dose)
			self.rx_list = rx_list

		except:
			print(str('Unknown error: prescription_data could not be '
					  'converted to conrad.Prescription() datatype.'))
			raise

	@property
	def list(self):
		""" List of structures in prescription """
		return self.rx_list

	@property
	def dict(self):
		""" Dictionary of structures in prescription, by label. """
		return {structure.label: structure for structure in self.rx_list}

	@property
	def constraints_by_label(self):
		"""
		Dictionary of constraints in prescription, by structure label.
		"""
		return self.constraint_dict

	def __str__(self):
		"""
		String of structures in prescription with attached constraints.
		"""
		return str(self.rx_list)

	def report(self, anatomy):
		"""
		Reports whether ``anatomy`` fulfills all prescribed constraints.

		Arguments:
			anatomy (:class:`Antomy`): Container of structures to
				compare against prescribed constraints.

		Returns:
			:obj:`dict`: Dictionary keyed by structure label, with data
			on each dose constraint associated with that structure in
			this :class:`Prescription`. Reported data includes the
			constraint, whether it was satisfied, and the actual dose
			achieved at the percentile/threshold specified by the
			constraint.

		Raises:
			TypeError: If ``anatomy`` not an :class:`Anatomy`.
		"""
		if not isinstance(anatomy, Anatomy):
			raise TypeError('argument "anatomy" must be of type{}'.format(
							Anatomy))

		rx_constraints = self.constraints_by_label
		report = {}
		for label, s in anatomy.structures.items():
			sat = []
			for constr in rx_constraints[label].itervalues():
				status, dose_achieved = s.satisfies(constr)
				sat.append({'constraint': constr, 'status': status,
							'dose_achieved': dose_achieved})
			report[label] = sat
		return report

	def report_string(self, anatomy):
		"""
		Reports whether ``anatomy`` fulfills all prescribed constraints.

		Arguments:
			anatomy (:class:`Anatomy`): Container of structures to
				compare against prescribed constraints.

		Returns:
			:obj:`str`: Stringified version of output from
			:attr:`Presription.report`.
		"""
		report = self.report(anatomy)
		out = ''
		for label, replist in report.items():
			sname = structures[label].name
			sname = '' if sname is None else ' ({})\n'.format(sname)
			for item in replist:
				out += str(
						'{}\tachieved? {}\tdose at level: {}\n'.format(
						 str(item['constraint']), item['status'],
						 item['dose_achieved']))
		return out
