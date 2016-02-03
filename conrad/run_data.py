class RunProfile(object):
	def __init__(self):
		pass
		# structures
		# rx doses
		# weights
		# dvh constraints
		# metadata: DVH slack, DVH 2pass

class RunOutput(object):
	def __init__(self):
		pass
		# x (beams), y (dose)
		# mu (dual var for x>= 0), nu (dual var for Ax = y)

class RunRecord(object):
	def __init__(self):
		self.profile = RunProfile()
		self.output = RunOutput()