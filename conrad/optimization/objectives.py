from conrad.compat import *

import abc
import numpy as np
import operator as op
import cvxpy

from conrad.defs import vec, module_installed
from conrad.physics.units import Gy, DeliveredDose
from conrad.physics.string import dose_from_string

WEIGHT_PWL_UNDER_DEFAULT = 1.
WEIGHT_PWL_OVER_DEFAULT = 0.05
WEIGHT_HINGE_DEFAULT = 1.
WEIGHT_LIN_NONTARGET_DEFAULT = 0.03

OPTKIT_INSTALLED = module_installed('optkit')
if OPTKIT_INSTALLED:
	import optkit as ok
	from optkit.libs.enums import OKFunctionEnums as fn_enums
else:
	ok = NotImplemented
	fn_enums = NotImplemented

@add_metaclass(abc.ABCMeta)
class TreatmentObjective(object):
	def __init__(self, **dose_and_weight_params):
		self.global_scaling = 1.
		self.normalization = 1.
		self.__weights = {}
		self.__doses = {}
		self.__aliases = {}
		self.__structure = None
		alias_dict = dose_and_weight_params.pop('aliases', {})
		for k, v in dose_and_weight_params.items():
			self.__setattr__(str(k), v)
		for attr, aliases in alias_dict.items():
			self.__add_aliases(attr, *aliases)

	def __getattr__(self, name):
		if not name.startswith('_'):
			raw = name.endswith('_raw')
			name = name.replace('_raw', '')

			if name in self.__aliases:
				name = self.__aliases[name]

			if 'weight' in name:
				if raw:
					normalization = 1.
				else:
					normalization = self.normalization
				if name in self.__weights:
					return normalization * self.__weights[name]
				else:
					raise AttributeError(
							'{} has no attribute {}'.format(type(self), name))
			elif 'dose' in name:
				if name in self.__doses:
					return self.__doses[name]
				else:
					raise AttributeError(
							'{} has no attribute {}'.format(type(self), name))

	def __setattr__(self, name, value):
		if self.__aliases is not None:
			if name in self.__aliases:
				name = self.__aliases[name]

		if 'weight' in name and not name.startswith('_'):
			weight = float(value)
			if weight < 0:
				raise ValueError('objective weight must be nonnegative')
			self.__weights[name] = weight
		elif 'dose' in name and not name.startswith('_'):
			if isinstance(value, str):
				dose = dose_from_string(value)
			else:
				dose = value
			if not isinstance(dose, DeliveredDose):
				raise TypeError(
						'objective dose argument `{}` must be of (or '
						'parsable as) type {}'
						''.format(name, DeliveredDose))
			else:
				self.__doses[name] = dose
		else:
			super(TreatmentObjective, self).__setattr__(name, value)

	def __add_aliases(self, attribute_name, *aliases):
		if attribute_name in self.__weights or attribute_name in self.__doses:
			for a in aliases:
				self.__aliases.update({str(a): attribute_name})

	def change_parameters(self, **parameters):
		for p in parameters:
			val = parameters[p]
			if p in self.__aliases:
				p = self.__aliases[p]
			if p in self.__doses or p in self.__weights:
				self.__setattr__(p, val)

	def scale(self, nonnegative_scalar):
		if float(nonnegative_scalar) >= 0:
			for k in self.__weights:
				self.__weights[k] *= nonnegative_scalar
			self.__last_scaling = float(nonnegative_scalar)
		else:
			raise ValueError('scaling must be nonnegative')

	def __mul__(self, other):
		return self.__imul__(other)

	def __rmul__(self, other):
		return self.__imul__(other)

	def __imul__(self, other):
		self.scale(other)
		return self

	def eval(self, y, voxel_weights=None):
		return self.primal_eval(y, voxel_weights)

	def expr(self, y_var, voxel_weights=None):
		return self.primal_expr(y_var, voxel_weights)

	def expr_Ax(self, A, x_var, voxel_weights=None):
		return self.primal_expr_Ax(A, x_var, voxel_weights)

	@abc.abstractproperty
	def is_target_objective(self):
		return NotImplementedError

	@abc.abstractproperty
	def is_nontarget_objective(self):
		return NotImplementedError

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
	def primal_expr_Ax(self, A, x_var, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def dual_expr(self, y_dual_var, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def dual_domain_constraints(self, nu_var, voxel_weights=None,
									 nu_offset=None, nonnegative=False):
		raise NotImplementedError

	@abc.abstractmethod
	def primal_expr_pogs(self, size, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def dual_expr_pogs(self, size, voxel_weights=None):
		raise NotImplementedError

	@abc.abstractmethod
	def dual_domain_constraints_pogs(self, size, voxel_weights=None,
									 nu_offset=None, nonnegative=False):
		raise NotImplementedError

	@property
	def parameters(self):
		p = {}
		p.update({k: str(self.__doses[k]) for k in self.__doses})
		p.update(self.__weights)
		return p

	@property
	def dict(self):
		return {
				'type': OBJECTIVE_TO_STRING[type(self)],
				'parameters': self.parameters
		}

	def string(self, offset=0):
		string = ''
		offset = int(offset) * '\t'
		string += offset + 'type: %s\n' %OBJECTIVE_TO_STRING[type(self)]
		string += offset + 'parameters:\n'
		offset += '\t'
		parameters = self.parameters
		for param in parameters:
			string += offset + param + ': ' + str(parameters[param]) + '\n'
		return string

	def __str__(self):
		return self.string()

class NontargetObjectiveLinear(TreatmentObjective):
	def __init__(self, weight=None, **options):
		if weight is None:
			weight = options.pop('w_over', WEIGHT_LIN_NONTARGET_DEFAULT)

		TreatmentObjective.__init__(self, weight=weight)
		self._TreatmentObjective__add_aliases(
				'weight', 'weight_overdose', 'weight_over', 'wt_over',
				'w_over', 'wt', 'w')

	@property
	def is_target_objective(self):
		return False

	@property
	def is_nontarget_objective(self):
		return True

	def primal_eval(self, y, voxel_weights=None):
		r"""
		Return :math:`c * \omega^T y`, for
		:math:`\omega\equiv```voxel_weights``
		"""
		if voxel_weights is None:
			return self.weight * np.sum(y)
		else:
			return self.weight * np.dot(voxel_weights, y)

	def dual_eval(self, nu, voxel_weights=None):
		""" Return ``0``"""
		return 0

	def primal_expr(self, y_var, voxel_weights=None):
		r"""
		Return :math:`c * \omega^T y`, for :math:`\omega\equiv```voxel_weights``
		"""
		if voxel_weights is None:
			return self.weight * cvxpy.sum_entries(y_var)
		else:
			return self.weight * cvxpy.sum_entries(
					cvxpy.mul_elemwise(voxel_weights, y_var))

	def primal_expr_Ax(self, A, x_var, voxel_weights=None):
		if voxel_weights is None:
			return self.weight * cvxpy.sum_entries(x_var.T * A.T)
		else:
			return self.weight * cvxpy.sum_entries(
					cvxpy.mul_elemwise(voxel_weights, (x_var.T * A.T).T))

	def dual_expr(self, nu_var, voxel_weights=None):
		""" Return ``0``"""
		return 0

	def dual_domain_constraints(self, nu_var, voxel_weights=None,
								nu_offset=None, nonnegative=False):
		r"""
		Return constraint :math:`\omega\nu = c`
		"""
		weight_vec = 1. if voxel_weights is None else voxel_weights
		offset = 0 if nu_offset is None else vec(nu_offset)
		constraints = [nu_var >= 0] if nonnegative else []
		constraints += [nu_var + offset == self.weight * weight_vec]
		return constraints

	def primal_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			weight_vec = 1. if voxel_weights is None else voxel_weights
			return ok.api.PogsObjective(
					size, h='Zero', c=0, d=weight_vec * self.weight)
		else:
			raise NotImplementedError

	def dual_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			return ok.api.PogsObjective(0)
		else:
			raise NotImplementedError

	def dual_domain_constraints_pogs(self, size, voxel_weights=None,
									 nu_offset=None, nonnegative=False):
		if OPTKIT_INSTALLED:
			raise NotImplementedError
		else:
			raise NotImplementedError

	def dual_fused_expr_constraints_pogs(self, size, voxel_weights=None,
									 nu_offset=None, nonnegative=False):
		"""
		simulatenously give dual expression

			:math:`f^*(\nu) = 0`

		and enforce dual domain constraints:

			:math:`\nu + \nu_{offset} == w`.

		if additionally desire :math:`nu \ge 0`, must have

			:math:`w  \ge \nu_{offset}`
		"""
		if OPTKIT_INSTALLED:
			weights = 1. if voxel_weights is None else vec(voxel_weights)
			offset = 0. if nu_offset is None else vec(nu_offset)
			return ok.api.PogsObjective(
					size, h='IndEq0', b=self.weight * weights - offset,
					d=-float(self.deadzone_dose) * weights)
		else:
			raise NotImplementedError

class TargetObjectivePWL(TreatmentObjective):
	def __init__(self, target_dose=None, weight_underdose=None,
				 weight_overdose=None, **options):
		if weight_underdose is None:
			weight_underdose = options.pop(
					'weight_under',
					options.pop('w_under', WEIGHT_PWL_UNDER_DEFAULT))
		if weight_overdose is None:
			weight_overdose = options.pop(
					'weight_over',
					options.pop('w_over', WEIGHT_PWL_OVER_DEFAULT))

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

	@property
	def is_target_objective(self):
		return True

	@property
	def is_nontarget_objective(self):
		return False

	def primal_eval(self, y, voxel_weights=None):
		r"""
		Return :math:`w_+ \omega^T(y-d)_+ + w_-\omega^T(y-d)_-`, for
		:math:`\omega \equiv` ``voxel weights``.
		"""
		residuals = vec(y) - float(self.target_dose)
		if voxel_weights is None:
			return float(
					self.weight_abs * np.linalg.norm(residuals, 1) +
					self.weight_linear * np.sum(residuals))
		else:
			return float(
					self.weight_abs * np.dot(voxel_weights, np.abs(residuals)) +
					self.weight_linear * np.dot(voxel_weights, residuals))

	def dual_eval(self, nu, voxel_weights=None):
		r"""
		Return :math:`-d^T\nu`
		"""
		if voxel_weights is None:
			return -float(self.target_dose) * np.sum(nu)
		else:
			return -float(self.target_dose) * np.dot(voxel_weights, nu)

	def primal_expr(self, y_var, voxel_weights=None):
		residuals = y_var.T - float(self.target_dose)
		if voxel_weights is not None:
			residuals = cvxpy.mul_elemwise(voxel_weights, residuals.T)
		return self.weight_abs * cvxpy.norm(residuals, 1) + \
			self.weight_linear * cvxpy.sum_entries(residuals)

	def primal_expr_Ax(self, A, x_var, voxel_weights=None):
		residuals = (x_var.T * A.T).T - float(self.target_dose)
		if voxel_weights is not None:
			residuals = cvxpy.mul_elemwise(voxel_weights, residuals)
		return self.weight_abs * cvxpy.norm(residuals, 1) + \
			self.weight_linear * cvxpy.sum_entries(residuals)

	def dual_expr(self, nu_var, voxel_weights=None):
		if voxel_weights is None:
			return -float(self.target_dose) * cvxpy.sum_entries(nu_var)
		else:
			return -float(self.target_dose) * cvxpy.sum_entries(
					cvxpy.mul_elemwise(voxel_weights, nu_var))

	def dual_domain_constraints(self, nu_var, voxel_weights=None,
								nu_offset=None):
		"""
		Return the constraint :math:`-w_- \le \nu \le w_+`.
		"""
		upper_bound = self.weight_overdose
		lower_bound = -self.weight_underdose
		voxel_weights = 1 if voxel_weights is None else vec(voxel_weights)
		offset = 0 if nu_offset is None else vec(nu_offset)
		constraints = [nu_var >= 0] if nonnegative else []
		constraints += [
				nu_var + offset <= voxel_weights * upper_bound,
				nu_var + offset >= voxel_weights * lower_bound
		]
		return constraints

	def primal_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			weights = 1. if voxel_weights is None else vec(voxel_weights)
			return ok.api.PogsObjective(
					size, h='Abs', b=float(self.target_dose),
					c=weights * self.weight_abs,
					d=weights * self.weight_linear)
		else:
			raise NotImplementedError

	def dual_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			weights = 1. if voxel_weights is None else vec(voxel_weights)
			return ok.api.PogsObjective(
					size, h='Zero', d=-float(self.target_dose) * weights)
		else:
			raise NotImplementedError

	def dual_domain_constraints_pogs(self, size, voxel_weights=None,
									 nu_offset=None, nonnegative=False):
		if OPTKIT_INSTALLED:
			raise NotImplementedError
		else:
			raise NotImplementedError

	def dual_fused_expr_constraints_pogs(self, size, voxel_weights=None,
									 nu_offset=None, nonnegative=False):
		"""
		simulatenously give dual expression

			f_conj(\nu) = -(dose * voxel_weights)^T * nu

		and enforce dual domain constraints:

			-w_- <= nu + nu_offset <= w_+.

		if additionally desire nu >= 0, instead enforce

			max(-(w_- + nu_offset), 0) <= nu <= max(w_+ - nu_offset, 0)
			L <= nu <= U

		rephrasing as a box constraint to the interval [0, 1], constrain
		each element nu_i to be:

			0 <= (nu_i - L) / (U - L) <= 1	; U - L > 0
			nu_i == 0						; U - L <= 0.

		"""
		if OPTKIT_INSTALLED:
			# f_conj = self.dual_expr_pogs(size, voxel_weights)
			# f_fused = self.dual_domain_constraint_pogs(
			# 		size, voxel_weights, nu_offset, nonnegative)
			# f_fused.set(d=f_conj.d)
			# return f_fused

			weights = 1. if voxel_weights is None else vec(voxel_weights)
			w_over = self.weight_overdose * voxel_weights
			w_under = self.weight_underdose * voxel_weights
			offset = 0. if nu_offset is None else vec(nu_offset)

			lower_limit = np.maximum(-(w_under + offset), 0)
			upper_limit = np.maximum(w_over - offset, 0)

			expr = __box01_pogs(size, lower_limit, upper_limit)
			expr.set(d=-float(self.deadzone_dose) * weights)
			return expr
		else:
			raise NotImplementedError

class ObjectiveHinge(TreatmentObjective):
	def __init__(self, deadzone_dose=None, weight=None, **options):
		if weight is None:
			weight = WEIGHT_HINGE_DEFAULT
		if deadzone_dose is None:
			deadzone_dose = options.pop('dose', 1 * Gy)
		TreatmentObjective.__init__(
				self, weight=weight, deadzone_dose=deadzone_dose)
		self._TreatmentObjective__add_aliases('deadzone_dose', 'dose')

	@property
	def is_target_objective(self):
		return True

	@property
	def is_nontarget_objective(self):
		return True

	def primal_eval(self, y, voxel_weights=None):
		residuals = vec(y) - float(self.deadzone_dose)
		residuals *= residuals > 0
		if voxel_weights is None:
			return self.weight * np.sum(residuals)
		else:
			return self.weight * np.dot(voxel_weights, residuals)

	def dual_eval(self, nu, voxel_weights=None):
		if voxel_weights is None:
			return -float(self.deadzone_dose) * np.sum(nu)
		else:
			return -float(self.deadzone_dose) * np.dot(voxel_weights, nu)

	def primal_expr(self, y_var, voxel_weights=None):
		residuals = cvxpy.pos(y_var.T - float(self.deadzone_dose))
		if voxel_weights is not None:
			residuals = cvxpy.mul_elemwise(voxel_weights, residuals.T)
		return self.weight * cvxpy.sum_entries(residuals)

	def primal_expr_Ax(self, A, x_var, voxel_weights=None):
		residuals = cvxpy.pos((x_var.T * A.T).T - float(self.deadzone_dose))
		if voxel_weights is not None:
			residuals = cvxpy.mul_elemwise(voxel_weights, residuals)
		return self.weight * cvxpy.sum_entries(residuals)

	def dual_expr(self, nu_var, voxel_weights=None):
		if voxel_weights is None:
			return -float(self.deadzone_dose) * cvxpy.sum_entries(nu_var)
		else:
			return -float(self.deadzone_dose) * cvxpy.sum_entries(
					cvxpy.mul_elemwise(voxel_weights, nu_var))

	def dual_domain_constraints(self, nu_var, voxel_weights=None,
									 nu_offset=None, nonnegative=False):
		"""
		Return the constraint :math:`0 \le \nu \le w`.
		"""
		voxel_weights = 1 if voxel_weights is None else vec(voxel_weights)
		offset = 0 if nu_offset is None else nu_offset
		constraints = [nu_var >= 0] if nonnegative else []
		constraints += [
			nu_var + offset <= voxel_weights * self.weight,
			nu_var + offset >= 0]
		return constraints

	def primal_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			weights = 1. if voxel_weights is None else vec(voxel_weights)
			return ok.api.PogsObjective(
					size, h='Abs', b=float(self.deadzone_dose),
					c=weights * self.weight / 2.,
					d=weights * self.weight / 2.)
		else:
			raise NotImplementedError

	def dual_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			weights = 1. if voxel_weights is None else vec(voxel_weights)
			return ok.api.PogsObjective(
					size, h='Zero', d=-float(self.deadzone_dose) * weights)
		else:
			raise NotImplementedError

	def dual_domain_constraints_pogs(self, size, voxel_weights=None,
									 nu_offset=None, nonnegative=False):
		if OPTKIT_INSTALLED:
			# weights = 1. if voxel_weights is None else vec(voxel_weights)
			# offset = 0. if nu_offset is None else vec(nu_offset)

			# lower_limit = np.maximum(-offset, 0)
			# upper_limit = np.maximum(weights - offset, 0)

			# return __box01_pogs(size, lower_limit, upper_limit)
			raise NotImplementedError
		else:
			raise NotImplementedError

	def dual_fused_expr_constraints_pogs(self, size, voxel_weights=None,
									 nu_offset=None, nonnegative=False):
		"""
		simulatenously give dual expression

			f_conj(\nu) = -(dose * voxel_weights)^T * nu

		and enforce dual domain constraints:

			0 <= nu + nu_offset <= w.

		if additionally desire nu >= 0, instead enforce

			max(-nu_offset, 0) <= nu <= max(w - nu_offset, 0)
			L <= nu <= U

		rephrasing as a box constraint to the interval [0, 1], constrain
		each element nu_i to be:

			0 <= (nu_i - L) / (U - L) <= 1	; U - L > 0
			nu_i == 0						; U - L <= 0.

		"""
		if OPTKIT_INSTALLED:
			# f_conj = self.dual_expr_pogs(size, voxel_weights)
			# f_fused = self.dual_domain_constraint_pogs(
			# 		size, voxel_weights, nu_offset, nonnegative)
			# f_fused.set(d=f_conj.d)
			# return f_fused

			weights = 1. if voxel_weights is None else vec(voxel_weights)
			offset = 0. if nu_offset is None else vec(nu_offset)

			lower_limit = np.maximum(-offset, 0)
			upper_limit = np.maximum(weights - offset, 0)

			expr = __box01_pogs(size, lower_limit, upper_limit)
			expr.set(d=-float(self.deadzone_dose) * weights)
			return expr
		else:
			raise NotImplementedError

def dictionary_to_objective(**options):
	if 'type' in options:
		return STRING_TO_OBJECTIVE[options.pop('type')](
				**options.pop('parameters', {}))
	else:
		raise ValueError('objective type not specified')

OBJECTIVE_TO_STRING = {
	NontargetObjectiveLinear: 'nontarget_linear',
	TargetObjectivePWL: 'target_piecewiselinear',
	ObjectiveHinge: 'hinge',
}
STRING_TO_OBJECTIVE = {v: k for k, v in OBJECTIVE_TO_STRING.items()}

# Auxilliary methods:
def __box01_pogs(size, lower_limit, upper_limit):
	if not OPTKIT_INSTALLED:
		raise RuntimeError('module `optkit` not installed')

	expr = ok.api.PogsObjective(size, h='IndBox01')
	U_minus_L = upper_limit - lower_limit
		h = expr.h
		a = expr.a
		b = expr.b
	for i in xrange(size):
		if U_minus_L > 0:
			a[i] = 1. / U_minus_L[i]
			b[i] = lower_limit[i] / U_minus_L[i]
		else:
			a[i] = 1.
			b[i] = 0.
			h[i] = fn_enums.dict['IndEq0']
	expr.set(h=h, a=a, b=b, c=1, d=0, e=0)