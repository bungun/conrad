"""
Defines methods for parsing dose constraint strings

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
from conrad.physics.units import percent, Gy, cGy, cm3, Gray, DeliveredDose
from conrad.physics.string import *
from conrad.medicine.dose.constraints import *

def v_strip(input_string):
	"""
	Strip 'v', 'V' and 'to' from input string.

	Preprocessing step for handling of string constraints of type
	"V20 Gy < 30 %" or "20 Gy to < 30%".
	"""
	return input_string.replace('to', '').replace('V', '').replace('v', '')

def d_strip(input_string):
	"""
	Strip 'd', and 'D' from input string.

	Preprocessing step for handling of string constraints of type
	"D70 < 20 Gy".
	"""
	return input_string.replace('D', '').replace('d', '')

def eval_constraint(string_constraint, rx_dose=None):
	"""
	Parse input string to form a new :class:`Constraint` instance.

	This method handles the following input cases.

	Absolute dose constraints:
		- "min > x Gy"
			- variants: "Min", "min"
			- meaning: minimum dose greater than x Gy

		- "mean < x Gy" ("mean > x Gy")
			- variants: "Mean, mean"
			- meaning: mean dose less than (more than) than x Gy

		- "max < x Gy"
			- variants: "Max", "max"
			- meaning: maximum dose less than x Gy

		- "D __ < x Gy" ("D __ > x Gy")
			- variants: "D __%", "d __%", "D __", "d __"
			- meaning: dose to __ percent of volume less than (greater than) x Gy

		- "V __ Gy < p %" ("V __ Gy > p %")
			- variants: "V __", "v __", "__ Gy to", "__ to"
			- meaning: no more than (at least) __ Gy to p percent of volume.

	Relative dose constraints:
		- "V __ %rx < p %" ("V __ %rx > p %")
			- variants: "V __%", "v __%", "V __", "v __"
			- meaning: at most (at least) p percent of structure receives  __ percent of rx dose.

		- "D __ < {frac} rx", "D __ > {frac} rx"
			- variants: "D __%", "d __%", "D __", "d __"
			- meaning: dose to __ percent of volume less than (greater than) frac * rx

	Absolute volume constraints:
		- "V __ Gy > x cm3" ("V __ Gy < x cm3"), "V __ rx > x cm3"  ("V __ rx < x cm3")
			- variants: "cc" vs. "cm3" vs. "cm^3"; "V __ _" vs. "v __ _"
			- error: convert to relative volume terms


	Arguments:
		string_constraint (:obj:`str`): Parsable string representation
			of dose constraint.
		rx_dose (:class:`DeliveredDose`, optional): Prescribed dose
			level to associate with dose constraint, required for
			relative dose constraints.

	Returns:
		:class:`Constraint`: Dose constraint specified by input.

	Raises:
		TypeError: If ``rx_dose`` not of type :class:`DeliveredDose`.
		ValueError: If input string specifies an absolute volume
			constraint, or if input is not well-formed (e.g., a dose
			quantity appears on LHS and RHS of inequality).
	"""
	string_constraint = str(string_constraint)

	if not isinstance(rx_dose, (type(None), DeliveredDose)):
		raise TypeError(
				'if provided, argument "rx_dose" must be of type {}, '
				'e.g., {} or {}'
				''.format(DeliveredDose, type(Gy), type(cGy)))

	if volume_unit_from_string(string_constraint) is not None:
		raise ValueError(
				'Detected dose volume constraint with absolute volume '
				'units. Convert to percentage.\n(input = {})'
				''.format(string_constraint))

	leq = '<' in string_constraint
	if leq:
		left, right = string_constraint.replace('=', '').split('<')
	else:
		left, right = string_constraint.replace('=', '').split('>')

	rdose = dose_unit_from_string(right) is not None
	ldose = dose_unit_from_string(left) is not None

	if rdose and ldose:
		raise ValueError(
				'Dose constraint cannot have a dose value on both '
				'sides of inequality.\n(input = {})'
				''.format(string_constraint))

	if rdose:
		tokens = ['mean', 'Mean', 'min', 'Min', 'max', 'Max', 'D', 'd']
		if not any(listmap(lambda t : t in left, tokens)):
			raise ValueError(
					'If dose specified on right side of inequality, '
					'left side must contain one of the following '
					'strings: \n.\ninput={}'
					''.format(tokens, string_constraint))

	relative = not rdose and not ldose
	relative &= rx_dose is not None

	if relative and (rdose or ldose):
		raise ValueError(
				'Dose constraint mixes relative and absolute volume '
				'constraint syntax. \n(input = {})'
				''.format(string_constraint))

	if not (rdose or ldose or relative):
		raise ValueError(
				'Dose constraint dose not specify a dose level in Gy '
				'or cGy, and no prescription\ndose was provided '
				'(argument "rx_dose") for parsing a relative dose '
				'constraint. \n(input = {})'
				''.format(string_constraint))

	try:
		# cases:
		# - "min > x Gy"
		# - "mean < x Gy"
		# - "max < x Gy"
		# - "D __% < x Gy"
		# - "D __% > x Gy"
		if rdose:
			#-----------------------------------------------------#
			# constraint in form "{LHS} <> {x} Gy"
			#
			# conversion to canonical form:
			#  (none required)
			#-----------------------------------------------------#

			# parse dose
			dose = dose_from_string(right)

			# parse threshold (min, mean, max or percentile)
			if 'mean' in left or 'Mean' in left:
				threshold = 'mean'
			elif 'min' in left or 'Min' in left:
				threshold = 'min'
			elif 'max' in left or 'Max' in left:
				threshold = 'max'
			else:
				threshold = percent_from_string(d_strip(left))

		# cases:
		# - "V __ Gy < p %" ( == "x Gy to < p %")
		# - "V __ Gy > p %" ( == "x Gy to > p %")
		elif ldose:
			#-----------------------------------------------------#
			# constraint in form "V{x} Gy <> {p} %"
			#
			# conversion to canonical form:
			# {x} Gy < {p} % ---> D{100 - p} < {x} Gy
			# {x} Gy > {p} % ---> D{p} > {x} Gy
			#-----------------------------------------------------#

			# parse dose
			dose = dose_from_string(v_strip(left))

			# parse percentile
			threshold = percent_from_string(right)

			# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			# VOLUME AT X GY < P % of STRUCTURE
			#
			# 	~equals~
			#
			# X Gy to < P% of structure
			#
			# 	~equivalent to~
			#
			# D(100 - P) < X Gy
			# <<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>
			# VOLUME AT X GY > P% of STRUCTURE
			#
			# 	~equals~
			#
			# X Gy to > P% of structure
			#
			# 	~equivalent to~
			#
			# D(P) > X Gy
			# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			if leq:
				threshold.value = 100 - threshold.value


		# cases:
		# - "V __% < p %"
		# - "V __% > p %"
		# - "D __% < {frac} rx"
		# - "D __% > {frac} rx"
		else:
			#-----------------------------------------------------#
			# constraint in form "V __% <> p%"
			#
			# conversion to canonical form:
			# V{x}% < {p} % ---> D{100 - p} < {x/100} * {rx_dose} Gy
			# V{x}% > {p} % ---> D{p} > {x/100} * {rx_dose} Gy
			#-----------------------------------------------------#
			if not 'rx' in right:
				# parse dose
				reldose = fraction_or_percent_from_string(
						v_strip(left.replace('rx', '')))
				dose = reldose * rx_dose

				# parse percentile
				threshold = percent_from_string(right)

				if leq:
					threshold.value = 100 - threshold.value

			#-----------------------------------------------------#
			# constraint in form "D{p}% <> {frac} rx" OR
			#					 "D{p}% <> {100 * frac}% rx"
			#
			# conversion to canonical form:
			# D{p}% < {frac} rx ---> D{p} < {frac} * {rx_dose} Gy
			# D{p}% >{frac} rx ---> D{p} > {frac} * {rx_dose} Gy
			#-----------------------------------------------------#
			else:
				# parse dose
				dose = fraction_or_percent_from_string(
						right.replace('rx', '')) * rx_dose

				# parse percentile
				threshold = percent_from_string(d_strip(left))

		if leq:
			return D(threshold) <= dose
		else:
			return D(threshold) >= dose

	except:
		print(str(
				'Unknown parsing error. Input = {}'.format(string_constraint)))
		raise