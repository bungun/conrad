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
	input_str = str(input_str)

	if any(listmap(lambda s: s in input_str, CM3_STRINGS)):
		return CM3()
	elif any(listmap(lambda s: s in input_str, MM3_STRINGS)):
		return MM3()
	else:
		return None

def dose_unit_from_string(input_str):
	input_str = str(input_str)

	if any(listmap(lambda s: s in input_str, CENTIGRAY_STRINGS)):
		return centiGray()
	elif any(listmap(lambda s: s in input_str, GRAY_STRINGS)):
		return Gray()
	else:
		return None

def strip_percent_units(input_str):
	input_str = str(input_str)

	output = input_str
	for s in PERCENT_STRINGS:
		output = output.replace(s, '')
	return output

def strip_volume_units(input_str):
	input_str = str(input_str)

	output = input_str
	for s in CM3_STRINGS + MM3_STRINGS:
		output = output.replace(s, '')
	return output

def strip_dose_units(input_str):
	input_str = str(input_str)

	output = input_str
	for s in CENTIGRAY_STRINGS + GRAY_STRINGS:
		output = output.replace(s, '')
	return output

def float_value_from_percent_string(input_str):
	return float(strip_percent_units(input_str))

def float_value_from_volume_string(input_str):
	return float(strip_volume_units(input_str))

def float_value_from_dose_string(input_str):
	return float(strip_dose_units(input_str))

def volume_from_string(input_str):
	unit = volume_unit_from_string(input_str)
	if unit is None:
		return None
	return float_value_from_volume_string(input_str) * unit

def dose_from_string(input_str):
	unit = dose_unit_from_string(input_str)
	if unit is None:
		return None
	return float_value_from_dose_string(input_str) * unit

def percent_from_string(input_str):
	return float_value_from_percent_string(input_str) * Percent()

def fraction_or_percent_from_string(input_str):
	if strip_percent_units(input_str) == input_str:
		return float(input_str)
	else:
		return percent_from_string(input_str)

def percent_or_dose_from_string(input_str):
	unit = dose_unit_from_string(input_str)
	if isinstance(unit, DeliveredDose):
		return dose_from_string(input_str)
	else:
		if strip_percent_units(input_str) != input_str:
			return percent_from_string(input_str)
		else:
			return None
