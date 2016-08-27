"""
Methods and constants for extracting physical units from strings.

Attributes:
	PERCENT_STRINGS (:obj:`list` of :obj:`str`): List of expected
		variants to test for when parsing strings for percentage units.
	CM3_STRINGS (:obj:`list` of :obj:`str`): List of expected variants
		to test for when parsing strings for cm^3 volume units.
	MM3_STRINGS (:obj:`list` of :obj:`str`): List of expected variants
		to test for when parsing strings for mm^3 volume units.
	GRAY_STRINGS (:obj:`list` of :obj:`str`): List of expected variants
		to test for when parsing strings for Gray dose units.
	CENTIGRAY_STRINGS (:obj:`list` of :obj:`str`): List of expected
		variants to test for when parsing strings for centiGray dose
		units.
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
from conrad.physics.units import Percent, MM3, CM3, Gray, centiGray, \
								 DeliveredDose

PERCENT_STRINGS = ['%', 'percent', 'Percent', 'PERCENT', 'pct', 'PCT']

CM3_STRINGS = ['cm3', 'cm^3', 'CM3', 'CM^3', 'cc', 'CC']
MM3_STRINGS = ['mm3', 'mm^3', 'MM3', 'MM^3']

GRAY_STRINGS = ['gray', 'Gray', 'GRAY', 'gy', 'Gy', 'GY']
CENTIGRAY_STRINGS = ['centigray', 'centiGray', 'cgy', 'cGy', 'cGY', 'CGY']
CENTIGRAY_STRINGS += ['centiGy', 'centiGRAY', 'CENTIGRAY']

def volume_unit_from_string(input_str):
	"""
	Parse volume unit from string.

	Arguments:
		input_str (:obj:`str`): String to parse.

	Returns:
		Instance of a :class:`Volume` object, namely :class:`CM3` if
		``input_str`` contains any of the entries of the list
		:attr:`CM3_STRINGS`, or :class:`MM3` if any substrings equal an
		entry from the list :attr:`MM3_STRINGS`. Returns ``None`` if no
		matches produced.
	"""
	input_str = str(input_str)

	if any(listmap(lambda s: s in input_str, CM3_STRINGS)):
		return CM3()
	elif any(listmap(lambda s: s in input_str, MM3_STRINGS)):
		return MM3()
	else:
		return None

def dose_unit_from_string(input_str):
	"""
	Parse dose unit from string.

	Arguments:
		input_str (:obj:`str`): String to parse.

	Returns:
		Instance of a :class:`DeliveredDose` object, namely
		:class:`Gray` if ``input_str`` contains any of the entries of
		the list :attr:`GRAY_STRINGS`, or `centiGray` if any substrings
		equal an entry from the list :attr:`CENTIGRAY_STRINGS`. Returns
		``None`` if no matches produced.
	"""
	input_str = str(input_str)

	if any(listmap(lambda s: s in input_str, CENTIGRAY_STRINGS)):
		return centiGray()
	elif any(listmap(lambda s: s in input_str, GRAY_STRINGS)):
		return Gray()
	else:
		return None

def strip_percent_units(input_str):
	"""
	Remove known percent unit representations from ``input_str``.

	Specifically, strip all members of :attr:`PERCENT_STRINGS` from
	``input_str``.
	"""
	input_str = str(input_str)

	output = input_str
	for s in PERCENT_STRINGS:
		output = output.replace(s, '')
	return output

def strip_volume_units(input_str):
	"""
	Remove known volume unit representations from ``input_str``.

	Specifically, strip all members of :attr:`CM3_STRINGS` and
	:attr:`MM3_STRINGS` from ``input_str``.
	"""
	input_str = str(input_str)

	output = input_str
	for s in CM3_STRINGS + MM3_STRINGS:
		output = output.replace(s, '')
	return output

def strip_dose_units(input_str):
	"""
	Remove known dose unit representations from ``input_str``.

	Specifically, strip all members of :attr:`CENTIGRAY_STRINGS` and
	:attr:`GRAY_STRINGS` from ``input_str``.
	"""
	input_str = str(input_str)

	output = input_str
	for s in CENTIGRAY_STRINGS + GRAY_STRINGS:
		output = output.replace(s, '')
	return output

def float_value_from_percent_string(input_str):
	"""
	Strip percent unit-related strings and convert remainder to :obj:`float`.
	"""
	return float(strip_percent_units(input_str))

def float_value_from_volume_string(input_str):
	"""
	Strip volume unit-related strings and convert remainder to :obj:`float`.
	"""
	return float(strip_volume_units(input_str))

def float_value_from_dose_string(input_str):
	"""
	Strip dose unit-related strings and convert remainder to :obj:`float`.
	"""
	return float(strip_dose_units(input_str))

def volume_from_string(input_str):
	""" Parse string as :class:`Volume`, extracting units and value. """
	unit = volume_unit_from_string(input_str)
	if unit is None:
		return None
	return float_value_from_volume_string(input_str) * unit

def dose_from_string(input_str):
	"""
	Parse string as :class:`DeliveredDose`, extracting units and value.
	"""
	unit = dose_unit_from_string(input_str)
	if unit is None:
		return None
	return float_value_from_dose_string(input_str) * unit

def percent_from_string(input_str):
	""" Parse string as :class:`Percent`, extracting units and value. """
	return float_value_from_percent_string(input_str) * Percent()

def fraction_or_percent_from_string(input_str):
	""" Parse string as :class:`Percent` or fraction (:obj:`float`). """
	if strip_percent_units(input_str) == input_str:
		return float(input_str)
	else:
		return percent_from_string(input_str)

def percent_or_dose_from_string(input_str):
	""" Parse string as :class:`Percent` or :class:`DeliveredDose`. """
	unit = dose_unit_from_string(input_str)
	if isinstance(unit, DeliveredDose):
		return dose_from_string(input_str)
	else:
		if strip_percent_units(input_str) != input_str:
			return percent_from_string(input_str)
		else:
			return None
