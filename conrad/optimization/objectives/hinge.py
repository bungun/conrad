from conrad.compat import *

from conrad.optimization.objectives.treatment import *

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
	def objective_type(self):
		return 'hinge'

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
		weight_vec = 1. if voxel_weights is None else vec(voxel_weights)
		offset = 0. if nu_offset is None else vec(nu_offset)
		constraints = [nu_var >= 0] if nonnegative else []
		constraints += [
				nu_var + offset <= weight_vec * self.weight,
				nu_var + offset >= 0]
		return constraints

	def primal_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			weights = 1. if voxel_weights is None else vec(voxel_weights)
			return ok.PogsObjective(
					size, h='Abs', b=float(self.deadzone_dose),
					c=weights * self.weight / 2.,
					d=weights * self.weight / 2.)
		else:
			raise NotImplementedError

	def dual_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			weights = 1. if voxel_weights is None else vec(voxel_weights)
			return ok.PogsObjective(
					size, h='Zero', d=-float(self.deadzone_dose) * weights)
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

			0 <= nu + nu_offset <= w.

		if additionally desire nu >= 0, instead enforce

			max(-nu_offset, 0) <= nu <= max(w - nu_offset, 0)
			L <= nu <= U

		rephrasing as a box constraint to the interval [0, 1], constrain
		each element nu_i to be:

			0 <= (nu_i - L) / (U - L) <= 1  ; U - L > 0
			nu_i == 0                                               ; U - L <= 0.

		"""
		if OPTKIT_INSTALLED:
			# f_conj = self.dual_expr_pogs(size, voxel_weights)
			# f_fused = self.dual_domain_constraint_pogs(
			#               size, voxel_weights, nu_offset, nonnegative)
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