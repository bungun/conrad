import abc
import numpy as np
import cvxpy

from conrad.physics.units import Gy, DeliveredDose

WEIGHT_UNDER_DEFAULT = 1
WEIGHT_OVER_DEFAULT = 0.05
WEIGHT_NONTARG_DEFAULT = 0.03

class TreatmentObjective(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self):
		self.__weights = []

	def scale(self, nonnegative_scalar):
		if float(nonnegative_scalar) >= 0:
			for w in self.__weights:
				w *= nonnegative_scalar
		else:
			raise ValueError('scaling must be nonnegative')

	def __rmul__(self, other):
		return self.scale(other)

	def __imul__(self, other):
		return self.scale(other)

	@abc.abstractmethod
	def primal_eval(self, y, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def dual_eval(self, y_dual, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def primal_expr(self, y_var, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def dual_expr(self, y_dual_var, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def dual_domain_constraints(self, nu_var, voxel_weights=None):
		raise NotImplementedError

class TargetObjective(TreatmentObjective):
	__metaclass__ = abc.ABCMeta

	def __init__(self, target_dose=None):
		TargetObjective.__init__(self)
		self.__dose = 1 * Gy

		if target_dose is not None:
			self.dose = target_dose

	@property
	def dose(self):
		return self.__dose

	@dose.setter
	def dose(self, dose):
		if isinstance(dose, DeliveredDose):
			self.__dose = dose
		else:
			raise TypeError('argument `dose` must be of type {}'.format(
					DeliveredDose))

	@abc.abstractmethod
	def primal_eval(self, y, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def dual_eval(self, nu, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def primal_expr(self, y_var, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def dual_expr(self, nu_var, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def dual_domain_constraints(self, nu_var, voxel_weights=None):
		raise NotImplementedError

class NontargetObjectiveLinear(TreatmentObjective):
	def __init__(self, weight=None):
		TreatmentObjective.__init__(self)

		self.__weight = WEIGHT_UNDER_DEFAULT
		self._TreatmentObjective__weights.append(self.__weight)

		if weight is not None:
			self.weight = weight

	@property
	def weight(self):
		return self.__weight

	@weight.setter
	def weight(self, weight):
		if weight < 0:
			raise ValueError('argument `weight` must be a nonnegative scalar')
		else:
			self.__weight = max(float(weight), 0)

	def primal_eval(self, y, voxel_weights=None):
		r"""
		Return :math:`c * \omega^T y`, for :math:`\omega\equiv```voxel_weights``
		"""
		if voxel_weights is None:
			return self.weight * np.sum(y_var)
		else:
			return self.weight * voxel_weights.dot(y_var)

	def dual_eval(self, nu, voxel_weights=None):
		""" Return ``0``"""
		return 0

	def primal_expr(self, y_var, voxel_weights=None):
		r"""
		Return :math:`c * \omega^T y`, for :math:`\omega\equiv```voxel_weights``
		"""
		if voxel_weights is None:
			return cvxpy.Expression(self.weight * cvxpy.sum(y_var))
		else:
			return cvxpy.Expression(self.weight * voxel_weights.dot(y_var))

	def dual_expr(self, nu_var, voxel_weights=None):
		""" Return ``0``"""
		return cvxpy.Expression(0)

	def dual_domain_constraints(self, nu_var, voxel_weights=None):
		r"""
		Return constraint :math:`\omega\nu = c`
		"""
		if voxel_weights is None:
			voxel_weights = 1

		return cvxpy.Constraint(nu_var == voxel_weights * self.weight)

class TargetObjectivePWL(TargetObjective):
	def __init__(self, target_dose=None, weight_underdose=None,
				 weight_overdose=None):
		TargetObjective.__init__(self, target_dose=target_dose)

		self.__weight_underdose = WEIGHT_UNDER_DEFAULT
		self.__weight_overdose = WEIGHT_OVER_DEFAULT
		self._TreatmentObjective__weights.append(self.__weight_underdose)
		self._TreatmentObjective__weights.append(self.__weight_overdose)

		if weight_underdose is not None:
			self.weight_underdose = weight_underdose
		if weight_overdose is not None:
			self.weight_overdose = weight_overdose

	@property
	def weight_underdose(self):
		return self.__weight

	@weight_underdose.setter
	def weight_underdose(self, weight):
		if weight < 0:
			raise ValueError('argument `weight` must be a nonnegative scalar')
		else:
			self.__weight_underdose = max(float(weight_underdose), 0)

	@property
	def weight_overdose(self):
		return self.__weight

	@weight_overdose.setter
	def weight_overdose(self, weight):
		if weight_overdose < 0:
			raise ValueError('argument `weight` must be a nonnegative scalar')
		else:
			self.__weight_overdose = max(float(weight_overdose), 0)

	@property
	def weight_abs(self):
		r"""
		Return :math:`b` such that :math:`b|y-d| + c(y-d) = w_+ (y - d)_+ +
		w_- (y-d)_-`.
		"""
		return 0.5 * (self.weight_overdose + self.weight_underdose)

	@property
	def weight_linear(self):
		r"""
		Return :math:`c` such that :math:`b|y-d| + c(y-d) = w_+ (y - d)_+
		+ w_- (y-d)_-`.
		"""
		return 0.5 * (self.weight_overdose - self.weight_underdose)

	def primal_eval(self, y, voxel_weights=None):
		r"""
		Return :math:`w_+ \omega^T(y-d)_+ + w_-\omega^T(y-d)_-`, for
		:math:`\omega \equiv` ``voxel weights``.
		"""
		residuals = y - self.dose
		if voxel_weights is None:
			return float(
					self.weight_abs * np.linalg.norm(residuals, 1) +
					self.weight_lin * np.sum(residuals))
		else:
			return float(
					self.weight_abs * voxel_weights.dot(np.abs(residuals)) +
					self.weight_lin * voxel_weights.dot(residuals))

	def dual_eval(self, nu, voxel_weights=None):
		r"""
		Return :math:`d^T\nu`
		"""
		return -self.dose * np.sum(nu)

	def primal_expr(self, y_var, voxel_weights=None):
		residuals = cvxpy.Expression(y_var - self.dose)
		if voxel_weights is None:
			return cvxpy.Expression(
					self.weight_abs * cvxpy.norm(residuals, 1) +
					self.weight_lin * cvxpy.sum(residuals))
		else:
			return cvxpy.Expression(
					self.weight_abs * voxel_weights.dot(cvxpy.abs(residuals)) +
					self.weight_lin * voxel_weights.dot(residuals))

	def dual_expr(self, nu_var, voxel_weights=None):
		return cvxpy.Expression(self.dose * cvxpy.sum(nu_var))

	def dual_domain_constraints(self, nu_var, voxel_weights=None):
		"""
		Return the constraint :math:`|\nu - c| \le b`.
		"""
		upper_bound = self.weight_lin + self.weight_abs
		lower_bound = self.weight_lin - self.weight_abs
		if voxel_weights is None:
			voxel_weights = 1
		return [cvxpy.Constraint(nu_var <= voxel_weights * upper_bound),
				cvxpy.Constraint(nu_var >= voxel_weights * lower_bound)]

class TargetObjectiveHinge(TreatmentObjective):
	def __init__(self, deadzone_dose=None, weight=None):
		TargetObjective.__init__(self, deadzone_dose)
		self.__weight = 0
		self.__deadzone = 0

		self.weight = weight
		self.deadzone_dose = deadzone_dose

	@property
	def weight(self):
		return self.__weight

	@weight.setter
	def weight(self, weight):
		if float(weight) < 0:
			raise ValueError('`weight` must be nonnegative')
		self.__weight = float(weight)

	@property
	def deadzone_dose(self):
		return self.__deadzone_dose

	@deadzone_dose.setter
	def deadzone_dose(self, deadzone_dose):
		if not isinstance(deadzone_dose, DeliveredDose):
			raise TypeError(
					'`deadzone_dose` must be of type {}'
					''.format(DeliveredDose))

	def primal_eval(self, y, voxel_weights=None):
		if '__iter__' in dir(y):
			if voxel_weights is not None:
				return np.sum()
			else:
				return np.sum()
		else:
			voxel_weight = 1. if voxel_weights is None else voxel_weights
			return self.weight * voxel_weight * max(y - self.deadzone_dose, 0)

	def dual_eval(self, nu, voxel_weights=None):
		return -self.dose * np.sum(nu)

	def primal_expr(self, y_var, voxel_weights=None):
		residuals = cvxpy.pos(y_var - self.deadzone_dose)
		if len(residual) == 1:
			voxel_weight = 1 if voxel_weights is None else float(voxel_weights)
			return cvxpy.Expression(self.weight * voxel_weight * residuals)
		elif voxel_weights is not None:
			return cvxpy.Expression(self.weight * voxel_weights.dot(residuals))
		else:
			return cvxpy.Expression(self.weight * cvxpy.sum(residuals))

	def dual_expr(self, nu_var, voxel_weights=None):
		return cvxpy.Expression(self.dose * cvxpy.sum(nu_var))

	def dual_domain_constraints(self, nu_var, voxel_weights=None):
		"""
		Return the constraint :math:`0 \le \nu \le 1`.
		"""
		if voxel_weights is None:
			voxel_weights = 1
		return [cvxpy.Constraint(nu_var <= voxel_weights * self.weight),
				cvxpy.Constraint(nu_var >= 0)]