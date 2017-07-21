# from conrad.compat import *

# from conrad.optimization.objectives.target_base import *

# class TargetObjectivePWQuadratic(TreatmentObjective):
# 	def primal_eval(self, y, voxel_weights=None):
# 		r"""
# 		Return :math:`0.5w_+ (y-d)_+^T diag(\omega^T) (y-d)_+ +
# 					  0.5w_-\omega^T(y-d)_- diag(\omega^T) \omega^T(y-d)_-`,
# 		for :math:`\omega \equiv` ``voxel weights``.
# 		"""
# 		residuals = vec(y) - float(self.target_dose)
# 		over = np.maximum(residuals, 0)
# 		under = over - residuals
# 		if voxel_weights is None:
# 			return 0.5 * float(
# 					self.weight_under * np.dot(under, under) +
# 					self.weight_over * np.dot(over, over))
# 		else:
# 			return 0.5 * float(
# 					self.weight_under * np.dot(under, voxel_weights * under) +
# 					self.weight_over * np.dot(over, voxel_weights * over))

# 	def dual_eval(self, y_dual, voxel_weights=None):
# 		over = np.maximum(vec(y_dual), 0)
# 		under = over - y_dual
# 		if voxel_weights is None:
# 			return 0.5 * float(
# 					np.dot(under, under) * 1. / self.weight_under +
# 					np.dot(over, over) * 1. / self.weight_over)
# 		else:
# 			return 0.5 * float(
# 					np.dot(under/voxel_weights, under) * 1. / self.weight_under
# 					+ np.dot(over/voxel_weights, over) * 1. / self.weight_over)

# 	def primal_expr(self, y_var, voxel_weights=None):
# 		residuals = y_var - float(self.target_dose)
# 		over = cvxpy.pos(residuals)
# 		under = cvxpy.neg(residuals)
# 		if voxel_weights is None:
# 			return 0.5 * (
# 			self.weight_over * over.T * over +
# 			self.weight_under * under.T * under)
# 		else:
# 			return 0.5 * (
# 			self.weight_over * over.T * cvxpy.mul_elemwise(voxel_weights, over) +
# 			self.weight_under * under.T * cvxpy.mul_elemwise(voxel_weights, under))

# 	def primal_expr_Ax(self, A, x_var, voxel_weights=None):
# 		return self.primal_expr(A * x_var, voxel_weights)

# 	def dual_expr(self, y_dual_var, voxel_weights=None):
# 		over = cvxpy.pos(y_dual_var)
# 		under = cvxpy.neg(y_dual_var)
# 		if voxel_weights is None:
# 			return 0.5 * (
# 				over.T * over / self.weight_over +
# 				under.T * under / self.weight_under
# 			)
# 		else:
# 			inv_wt = 1. / voxel_weights
# 			return 0.5 * (
# 				over.T * cvxpy.mul_elemwise(inv_wt, over) / self.weight_over +
# 				under.T * cvxpy.mul_elemwise(inv_wt, under) / self.weight_under
# 			)

# 	def dual_domain_constraints(self, nu_var, voxel_weights=None,
#								 nu_offset=None, nonnegative=False):
# 		return []

# 	def primal_expr_pogs(self, size, voxel_weights=None):
# 		weights = 1. if voxel_weights is None else vec(voxel_weights)
# 		return ok.PogsObjective(
# 				size, h='AsymmQuad', b=float(self.target_dose),
# 				c=weights * self.weight_under,
# 				s=self.weight_over / self.weight_under)

# 	def dual_expr_pogs(self, size, voxel_weights=None):
# 		weights = 1. if voxel_weights is None else 1./vec(voxel_weights)
# 		return ok.PogsObjective(
# 				size, h='AsymmQuad', c=weights / self.weight_under,
# 				s=self.weight_under / self.weight_over)


# 	def dual_domain_constraints_pogs(self, size, voxel_weights=None,
#									 nu_offset=None, nonnegative=False):
# 		raise NotImplementedError

#	def dual_fused_expr_constraints_pogs(self, structure, ,
#										 nu_offset=None, nonnegative=False):
#			raise NotImplementedError