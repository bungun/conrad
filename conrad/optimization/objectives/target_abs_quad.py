# from conrad.compat import *

# from conrad.optimization.objectives.target_base import *

# class TargetObjectiveAbsQuad(TreatmentObjective):
# 	def primal_eval(self, y, voxel_weights=None):
# 		residuals = vec(y) - float(self.target_dose)
# 		over = np.maximum(residuals, 0)
# 		under = over - residuals
# 		if voxel_weights is None:
# 			return float(
# 					self.w_under * np.sum(-under) +
# 					0.5 * self.w_over * np.dot(over, over))
# 		else:
# 			return float(
# 					self.w_under * np.dot(voxel_weights, -under) +
# 					0.5 * self.w_over * np.dot(over, voxel_weights * over)
# 			)

# 	def dual_eval(self, y_dual, voxel_weights=None):
# 		residuals = vec(y) - float(self.target_dose)
# 		over = np.maximum(residuals, 1)
# 		if voxel_weights is None:
# 			return float(0.)
# 		else:
# 			return float(0.)

# 	def primal_expr(self, y_var, voxel_weights=None):
# 		residuals =

# 	def primal_expr_Ax(self, A, x_var, voxel_weights=None):
# 		return self.primal_expr(A * x_var, voxel_weights)

# 	def dual_expr(self, y_dual_var, voxel_weights=None):
# 		return

# 	def dual_domain_constraints(self, nu_var, voxel_weights=None):
# 		return

# 	def primal_expr_pogs(self, size, voxel_weights=None):
# 		if OPTKIT_INSTALLED:
# 			weights = 1. if voxel_weights is None else vec(voxel_weights)
# 			return okPogsObjective(
# 					size, h='AbsQuad', c=)
# 		else:
# 			raise NotImplementedError
# 		return PogsObjective(
# 			)

# 	def dual_expr_pogs(self, size, voxel_weights=None):
# 		raise NotImplementedError

# 	def dual_domain_constraints_pogs(self, size, voxel_weights=None):
# 		raise NotImplementedError