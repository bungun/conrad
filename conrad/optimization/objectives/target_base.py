from conrad.compat import *

from conrad.optimization.objectives.treatment import *

@add_metaclass(abc.ABCMeta)
class TargetObjectiveTwoSided(TreatmentObjective):
	def __init__(self, target_dose=None, weight_underdose=None,
				 weight_overdose=None, **options):
		if weight_underdose is None:
			weight_underdose = options.pop(
					'weight_under',
					options.pop('w_under', options.pop(
							'default_underdose_weight',
							WEIGHT_PWL_UNDER_DEFAULT)))

		if weight_overdose is None:
			weight_overdose = options.pop(
					'weight_over',
					options.pop('w_over', options.pop(
							'default_overdose_weight',
							WEIGHT_PWL_OVER_DEFAULT)))

		if target_dose is None:
			target_dose = options.pop('dose', 1 * Gy)
		TreatmentObjective.__init__(
				self, weight_underdose=weight_underdose,
				weight_overdose=weight_overdose, target_dose=target_dose)
		self._TreatmentObjective__add_aliases(
				'weight_underdose', 'weight_under', 'w_under')
		self._TreatmentObjective__add_aliases(
				'weight_overdose', 'weight_over', 'w_over')
		self._TreatmentObjective__add_aliases('target_dose', 'dose')

	@property
	def is_target_objective(self):
		return True

	@property
	def is_nontarget_objective(self):
		return False