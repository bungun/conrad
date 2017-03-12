from conrad.compat import *

from conrad.optimization.objectives.treatment import TreatmentObjective
from conrad.optimization.objectives.nontarget_linear import NontargetObjectiveLinear
from conrad.optimization.objectives.target_pwl import TargetObjectivePWL
# from conrad.optimization.objectives.target_pwq import TargetObjectivePWQ
# from conrad.optimization.objectives.target_abs_quad import TargetObjectiveAbsQuad
# from conrad.optimization.objectives.target_abs_exp import TargetObjectiveAbsExp
# from conrad.optimization.objectives.target_huber import TargetObjectiveBerhu
# from conrad.optimization.objectives.target_berhu import TargetObjectiveHuber
from conrad.optimization.objectives.hinge import ObjectiveHinge

OBJECTIVES = [
	NontargetObjectiveLinear, TargetObjectivePWL, #TargetObjectivePWQ,
	# TargetObjectiveHuber, TargetObjectiveBerhu, TargetObjectiveAbsExp,
	# TargetObjectiveAbsQuad,
	ObjectiveHinge,
]

STRING_TO_OBJECTIVE = {o().objective_type: o for o in OBJECTIVES}


def dictionary_to_objective(**options):
	if 'type' in options:
		return STRING_TO_OBJECTIVE[options.pop('type')](
				**options.pop('parameters', {}))
	else:
		raise ValueError('objective type not specified')