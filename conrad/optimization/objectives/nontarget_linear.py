from conrad.compat import *

from conrad.optimization.objectives.treatment import *

class NontargetObjectiveLinear(TreatmentObjective):
	def __init__(self, weight=None, **options):
		if weight is None:
			weight = options.pop('w_over', WEIGHT_LIN_NONTARGET_DEFAULT)

		TreatmentObjective.__init__(self, weight=weight)
		self._TreatmentObjective__add_aliases(
				'weight', 'weight_overdose', 'weight_over', 'wt_over',
				'w_over', 'wt', 'w')

	@property
	def objective_type(self):
		return 'nontarget_linear'

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

	def dual_domain_constraints(self, nu_var, voxel_weights=None):
		r"""
		Return constraint :math:`\omega\nu = c`
		"""
		weight_vec = 1. if voxel_weights is None else voxel_weights
		return nu_var == self.weight * weight_vec

	def primal_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			weight_vec = 1. if voxel_weights is None else voxel_weights
			return ok.PogsObjective(
					size, h='Zero', c=0, d=weight_vec * self.weight)
		else:
			raise NotImplementedError

	def dual_expr_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			return ok.PogsObjective(0)
		else:
			raise NotImplementedError

	def dual_domain_constraints_pogs(self, size, voxel_weights=None):
		if OPTKIT_INSTALLED:
			raise NotImplementedError
		else:
			raise NotImplementedError