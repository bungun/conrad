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

	def dual_domain_constraints(self, nu_var, voxel_weights=None):
		"""
		Return the constraint :math:`0 \le \nu \le w`.
		"""
		if voxel_weights is None:
			voxel_weights = 1
		else:
			voxel_weights = vec(voxel_weights)
		return [nu_var <= voxel_weights * self.weight, nu_var >= 0]

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

	def dual_domain_constraints_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			raise NotImplementedError
		else:
			raise NotImplementedError