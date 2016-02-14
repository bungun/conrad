from conrad.structure import Structure
from conrad.dose import D
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
  - "D90 >= 32.3Gy"
  - "D1 <= 1.1rx"

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
		'constraints' : ['D1 <= 1.1rx', 'D90 >= 32.3Gy']
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


def string2constraint(string_constraint, rx_dose=None):
	""" TODO: docstring

	convert input string to standard form dose constraint:

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
		meaning: dose to __ percent of volume less than (greater than) x Gy
	

	- "x Gy to < p %"
	- "x Gy to > p %"
	
		meaning: no more than (at least) x Gy to p percent of volume


	relative dose constraints
	-------------------------

	- "V__ < p %"
	- "V__ > p %"

		variants: "V__%", "v__%", "V__", "v__"
		meaning: volume receiving __ percent of rx dose less than 
			(greater than) p percent of structure volume
	

	 - "D__ < {frac} rx"
	 - "D__ > {frac} rx"

		variants: "D__%", "d__%", "D__", "d__"
		meaning: dose at to __ percent of volume less than (greater than)
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
		left, right = string_constraint.strip('=').split('<')
	else:
		left, right = string_constraint.strip('=').split('>')

	rdose = 'Gy' in right
	ldose = 'Gy' in left 

	if rdose and ldose:
		ValueError("Dose constraint cannot have "
			"a dose value on both sides of inequality."
			"\n(input = {})".format(string_constraint))
	
	if rdose:
		tokens = ['mean', 'Mean', 'min', 'Min', 'max', 'Max', 'D', 'd']
		if not any(map(lambda t : t in left, tokens)):
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
			#-----------------------------------------------------#
			# constraint in form "{LHS} <> {x} Gy"
			#
			# conversion to canonical form:
			# -if LHS == "D__" : (none required).
			# -if LHS == "min", "mean", "max" : 
			#	convert to D0, D50, D100, respectively.
			# (inequality direction preserved)
			#-----------------------------------------------------#

			# parse dose
			if 'cGy' in right:
				dose = float(right.strip('cGy')) / 100.
			else:
				dose = float(right.strip('Gy'))
		
			# parse percentile
			if 'mean' in left or 'Mean' in left:
				percentile = 50.
			elif 'min' in left or 'Min' in left:
				percentile = 100.
			elif 'max' in left or 'Max' in left:
				percentile = 0.
			else:
				percentile = float(left.strip('%').strip('d').strip('D'))

			# parse direction:
			# dose on right-hand side of inequality is same as canonical
			# form, so parsed inequality direction = input inequality direction
			direction = '<' if lt else '>'

		# cases: 
		# - "x Gy to < p %"
		# - "x Gy to > p %"
		elif ldose:
			#-----------------------------------------------------#
			# constraint in form "{x} Gy <> {p} %"
			#
			# conversion to canonical form:
			# {x} Gy < {p} % ---> D{p} > {x} Gy
			# {x} Gy > {p} % ---> D{p} < {x} Gy
			# (inequality direction flips)
			#-----------------------------------------------------#

			# parse dose
			if 'cGy' in left:
				dose = float(left.strip('cGy')) / 100.
			else:
				dose = float(left.strip('Gy'))

			# parse percentile
			percentile = float(right.strip('%'))

			# parse direction
			direction = '>' if lt else '<'

		# cases: 
		# - "V__% < p %"
		# - "V__% > p %"
		# - "D__% < {frac} rx"
		# - "D__% > {frac} rx"
		else:
			#-----------------------------------------------------#
			# constraint in form "V__% <> p%"
			#
			# conversion to canonical form:
			# V{x}% < {p} % ---> D{p} > {x/100} * {rx_dose} Gy
			# V{x}% > {p} % ---> D{p} < {x/100} * {rx_dose} Gy
			# (inequality direction flips)
			#-----------------------------------------------------#
			if not 'rx' in right:
				# parse dose
				reldose = float(left.strip('%').strip('v').strip('V'))
				dose = reldose / 100. * rx_dose

				# parse percentile
				percentile = float(right.strip('%'))

				# parse direction:
				direction = '>' if lt else '<'

			#-----------------------------------------------------#
			# constraint in form "D__% <> {frac} rx"
			#
			# conversion to canonical form:
			# D{p}% < {frac} rx ---> D{p} < {frac} * {rx_dose} Gy
			# D{p}% >{frac} rx ---> D{p} > {frac} * {rx_dose} Gy
			# (inequality direction preserved)
			#-----------------------------------------------------#
			else:
				# parse dose
				dose = rx_dose * float(right.strip('rx'))

				# parse percentile
				percentile = float(left.strip('%').strip('d').strip('D'))

				# parse direction:
				# 
				direction = '<' if lt else '>'


		if '<' in direction:
			return D(percentile) <= dose
		else:
			return D(percentile) >= dose

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
		self.rx_list = []
		if prescription_data is not None:
			self.digest(prescription_data) 


	def digest(self, prescription_data):
		""" TODO: docstring """
		
		err = None
		data_valid = False
		rx_list = []
		if isinstance(prescription_data, list):
			rx_list = prescription_data
			data_valid = True

		if isinstance(prescription_data, str):
			if path.exists(prescription_data):
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
			raise TypeError("input prescription_data expected to be "
							"a list or the path to a valid JSON or YAML file.")
							
		try:
			for item in rx_list:
				label = item['label']
				self.structure_dict[label] = Structure(
					label = item['label'],
					name = item['name'],
					is_target = bool(item['is_target']), 
					dose = float(item['dose']) if item['dose'] is not None else 0.)
				self.constraint_dict[label] = ConstraintList()

				if item['constraints'] is not None:
					[self.constraint_dict[label] += string2constraint(s) for s in item['constraints']]


			self.rx_list = rx_list

		except:
			print str("Unknown error: prescription_data could not be "
				"converted to conrad.Prescription() datatype.") 
			raise

	@property
	def list(self):
		""" TODO: docstring """
		return self.rx_list
	
	@property
	def dict(self):
		""" TODO: docstring """
		rx_dict = {}
		for structure in self.rx_list:
			rx_dict[structure.label] = structure
		return self.rx_dict

	@property
	def constraints_by_label(self):
		""" TODO: docstring """
		return self.constraint_dict

	def __str__(self):
		""" TODO: docstring """
		return str(self.rx_list)

def report(self, structures):
	"""TODO: docstring"""
	rx_constraints = self.constraints_by_label
	report = {}
	for label, s in structures.iteritems():
		sat = []
		for constr in rx_constraints[label]:
			status, dose_achieved = s.satisfies(constr)
			sat.append({'constraint': constr, 
				'status': status, 'dose_achieved': dose_achieved})
		report[label] = sat
	return report

@staticmethod
def report_string(self, structures):
	report = self.report(structures)
	out = ''
	for label, replist in report.iteritems():
		sname = structures[label].name
		sname = '' if sname is None else ' ({})\n'.format(sname)
		for item in replist:
			out += '{}\tachieved? {}\tdose at level: {}\n'.format(
				str(item['constraint']),
				item['status'], 
				item['dose_achieved'])



