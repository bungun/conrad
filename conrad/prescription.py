from conrad.dvh import canonical_string_to_tuple
from conrad.structure import Structure
from os import path
import json, yaml
from traceback import format_exc

# TODO: unit test
"""
TODO: prescription.py docstring

Parsing methods expect following format:

YAML:
=====

- name : PTV
  label : 1
  is_target: Yes
  dose : 35.
  constraints:
  - "D99 <= 1.1rx"
  - "D20 >= 10.1Gy"
  - "D90 >= 32.3Gy"

- name : OAR1
  label : 2
  is_target: No
  dose : 
  constraints:
  - "D95 <= 20Gy"



Python list of dictionaries (JSON approximately same)
=====================================================

	[{
		'name' : 'PTV',
		'label' : 1,
		'is_target' : True,
		'dose' : 35.,
		'constraints' : ['D99 <= 1.1rx', 'D20 >= 10.1Gy', 'D90 >= 32.3Gy']
	},

	{
 		'name' : 'OAR1',
	  	'label' : 2,
	  	'is_target' : False,
	  	'dose' : None,
	  	'constraints' : ['D95 <= 20Gy']
	}]


( JSON differences: 
	- double quote instead of single
	- true/false instead of True/False
	- null instead of None )
"""


def canonicalize_dvhstring(string_constraint, rx_dose=None):
	""" TODO: docstring

	convert input string to standard form dose constraint 
	string:

		D{p} <= {d} Gy (dose @ pth percentile <= d Gy)

	or

		D{p} >= {d} Gy (dose @ pth percentile >= d Gy)


	input cases:

	absolute dose constraints
	-------------------------
	
	- "min > x Gy"
	
		variants: "Min", "min"	
		meaning: minimum dose no less than x Gy
	
	
	- "mean < x Gy"

		variants: "Mean, mean"
		meaning: mean dose no more than x Gy
	

	- "max < x Gy"

		variants: "Max", "max"
		meaning: maximum dose no more than x Gy

	
	- "D__ < x Gy"
	- "D__ > x Gy"
	
		variants: "D__%", "d__%", "D__", "d__"
		meaning: dose at __th percentile less than (greater than) x Gy
	

	- "x Gy to < p %"
	- "x Gy to > p %"
	
		meaning: no more than (at least) x Gy to p percent of volume


	relative dose constraints
	-------------------------

	- "V__ < p %"
	- "V__ > p %"

		variants: "V__%", "v__%", "V__", "v__"
		meaning: volume at __ percent of rx dose less than 
			(greater than) p percent of volume
	

	 - "D__ < {frac} rx"
	 - "D__ > {frac} rx"

		variants: "D__%", "d__%", "D__", "d__"
		meaning: dose at __th percentile less than (greater than)
			frac * rx 


	absolute volume constraints:
	----------------------------
	- "volume @ b Gy < x cc"
	- "volume @ b Gy < x cm3"
		
		error: convert to relative volume terms
	"""


	for token in ['cm3', 'cc', 'CC', 'cm^3']:
		if token in string_constraint:
			ValueError("Detected dose volume constraint with "
			"absolute volume units. Convert to percentage."
			"\n(input = {})".format(string_constraint))

	lt = '<' in string_constraint

	if lt:
		left, right = string_constraint.split('<').strip('=')
	else:
		left, right = string_constraint.split('>').strip('=')

	rdose = 'Gy' in right
	ldose = 'Gy' in left 

	if rdose and ldose:
		ValueError("Dose constraint cannot have "
			"a dose value on both sides of inequality."
			"\n(input = {})".format(string_constraint))
	
	if rdose:
		tokens = ['mean', 'Mean', 'min', 'Min', 'max', 'Max', 'D', 'd']
		if not any(map(t in left, tokens)):
			ValueError("If dose specified on right side "
				"of inequality, left side must contain one "
				" of the following strings: \n.\ninput={}".format(
					tokens, string_constraint))

	relative = 'v' in left 
	relative |= 'V' in left
	relative |= 'd' in left and 'rx' in right
	relative |= 'D' in left and 'rx' in right
	relative &= rx_dose is not None

	if relative and (rdose or ldose):
		ValueError("Dose constraint mixes relative "
			"and absolution volume constraint syntax."
			"\n(input = {})".format(string_constraint))

	if not rdose or ldose or relative:
		ValueError("Dose constraint dose not "
			"specify a dose level in Gy or cGy, "
			"and no prescription dose was provided "
			"(argument rx_dose) for parsing "
			"a relative dose constraint."
			"\n(input = {})".format(string_constraint))


	try:
		# cases:
		# - "min > x Gy"
		# - "mean < x Gy"
		# - "max < x Gy"
		# - "D__% < x Gy"
		# - "D__% > x Gy"
		if rdose:
			# parse dose
			if 'cGy' in right:
				dvhc['dose'] = float(right.strip('cGy')) / 100.
			else:
				dvhc['dose'] = float(right.strip('Gy'))
		
			# parse percentile
			if 'mean' in left or 'Mean' in left:
				dvhc['percentile'] = 50.
			elif 'min' in left or 'Min' in left:
				dvhc['percentile'] = 0.
			elif 'max' in left or 'Max' in left:
				dvhc['percentile'] = 100.
			else:
				dvhc['percentile'] = float(left.strip('%').strip('d').strip('D'))

			# TODO: parse direction

		# cases: 
		# - "x Gy to < p %"
		# - "x Gy to > p %"
		elif ldose:
			# parse dose
			if 'cGy' in left:
				dvhc['dose'] = float(left.strip('cGy')) / 100.
			else:
				dvhc['dose'] = float(left.strip('Gy'))

			# parse percentile
			dvhc['percentile'] = float(right.strip('%'))

			# TODO: parse direction

		# cases: 
		# - "V__% < p %"
		# - "V__% > p %"
		# - "D__% < {frac} rx"
		# - "D__% > {frac} rx"
		else:
			if not 'rx' in right:
				# parse dose
				reldose = float(left.strip('%').strip('v').strip('V'))
				dvhc['dose'] = reldose / 100. * rx_dose

				# parse percentile
				dvhc['percentile'] = float(right.strip('%'))
			else:
				# parse dose
				dvhc['dose'] = rx_dose * float(right.strip('rx'))

				# parse percentile
				dvhc['percentile'] = float(left.strip('%').strip('d').strip('D'))


			# TODO: parse direction


		# cases:
		# x Gy to < > p%
		elif 'Gy' in left:
			dvhc['dose'] = float(left.strip('cGy')) / 100.
			left_parsed = True
			dose_parsed = True

		elif 'Gy' in right:
			dvhc['dose'] = float(left.strip('Gy'))
			left_parsed = True
			dose_parsed = True

		elif rx_dose is None:
			ValueError("Dose constraint does not "
				"specify a ")
	except:
		print str("Unknown parsing error. Input = {}".format(
			string_constraint))
		raise




class Prescription(object):
	""" TODO: docstring """

	def __init__(self, prescription_data = None):
		""" TODO: docstring """
		self.constraint_dict = {}
		self.structure_dict = {}
		if prescription_data is not None:
			self.digest(prescription_data) 


	def digest(self, prescription_data):
		""" TODO: docstring """

		err = None
		data_valid = False
		if isinstance(prescription_data, list):
			rx_list = prescription_data
			data_valid = True

		if isinstance(data_valid, str):
			if path.isfile(prescription_data):
				try:
					f = open(prescription_data)
					if '.json' in prescription_data:
						rx_list = json.load(f)
					else:
						rx_list = yaml.safe_load(f)
					f.close
					data_valid = True

				except:
					err = format_exc()

		if not data_valid:
			if err is not None: print err 
			TypeError("input prescription_data expected to be "
				"a list or the path to a valid JSON or YAML file.")

		try:
			for item in rx_list:
				label = item['label']
				self.structure_dict[label] = Structure(
					label = item['label'],
					name = item['name'],
					is_target = bool(item['is_target']), 
					dose = float(item['dose']))
				self.constraint_dict[label] = []
				for string in item['constraints']:
					constr = canonicalize_dvhstring(string)
					self.constraint_dict[label].append(
						canonical_string_to_tuple(constr))
		except:
			print str("Unknown error: prescription_data could not be "
				"converted to conrad.Prescription() datatype.") 
			raise


	def convert mean_dose_2_dvh(self,)