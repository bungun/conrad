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

	def build_dual_domain_constraints(self, voxel_weights):
		"""
		Append the constraints :math:`0 \le \nu \le w` to
		:attr:`TreatmentObjective.dual_constraint_queue` for use in
		:meth:`TreatmentObjective.satisfies_dual_domain_constraints`.
		"""
		self.dual_constraint_queue.enqueue('==', self.weight * voxel_weights)

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
		weight_vec = 1. if voxel_weights is None else vec(voxel_weights)
		offset = 0. if nu_offset is None else vec(nu_offset)
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
			return ok.api.PogsObjective(size, h='Zero')
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
			)
		else:
		        raise NotImplementedError