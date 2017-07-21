from conrad.compat import *

from conrad.optimization.objectives.target_base import *

class TargetObjectivePWL(TargetObjectiveTwoSided):
	def __init__(self, target_dose=None, weight_underdose=None,
				 weight_overdose=None, **options):
		TargetObjectiveTwoSided.__init__(
				self, target_dose, weight_underdose, weight_overdose,
				default_underdose_weight=WEIGHT_PWL_UNDER_DEFAULT,
				default_overdose_weight=WEIGHT_PWL_OVER_DEFAULT, **options)

	@property
	def objective_type(self):
		return 'target_piecewise_linear'

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
								nu_offset=None, nonnegative=False):
		"""
		Return the constraint :math:`-w_- \le \nu \le w_+`.
		"""
		upper_bound = self.weight_overdose
		lower_bound = -self.weight_underdose

		weight_vec = 1. if voxel_weights is None else vec(voxel_weights)
		offset = 0. if nu_offset is None else vec(nu_offset)
		constraints = [nu_var >= 0] if nonnegative else []
		constraints += [
				nu_var + offset <= weight_vec * upper_bound,
				nu_var + offset >= weight_vec * lower_bound
		]
		return constraints

	def primal_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			weights = 1. if voxel_weights is None else vec(voxel_weights)
			return ok.PogsObjective(
					size, h='Abs', b=float(self.target_dose),
					c=weights * self.weight_abs,
					d=weights * self.weight_linear)
		else:
			raise NotImplementedError

	def dual_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			weights = 1. if voxel_weights is None else vec(voxel_weights)
			return ok.PogsObjective(
					size, h='Zero', c=-float(self.target_dose) * weights)
		else:
			raise NotImplementedError

	def dual_domain_constraints_pogs(self, size, voxel_weights=None,
									 nu_offset=None, nonnegative=False):
		if OPTKIT_INSTALLED:
			raise NotImplementedError
		else:
			raise NotImplementedError

	def dual_fused_expr_constraints_pogs(self, structure, voxel_weights=None,
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