# from conrad.compat import *

# from conrad.optimization.objectives.target_base import *

# class TargetObjectiveAbsExp(TreatmentObjective):
# 	def primal_eval(self, y, voxel_weights=None):
# 		raise NotImplementedError

# 	def dual_eval(self, y_dual, voxel_weights=None):
# 		raise NotImplementedError

# 	def primal_expr(self, y_var, voxel_weights=None):
# 		raise NotImplementedError

# 	def primal_expr_Ax(self, A, x_var, voxel_weights=None):
# 		raise NotImplementedError

# 	def dual_expr(self, y_dual_var, voxel_weights=None):
# 		raise NotImplementedError

# 	def dual_domain_constraints(self, nu_var, voxel_weights=None,
#								nu_offset=None, nonnegative=False):
# 		raise NotImplementedError

# 	def primal_expr_pogs(self, size, voxel_weights=None):
# 		raise NotImplementedError

# 	def dual_expr_pogs(self, size, voxel_weights=None):
# 		raise NotImplementedError

# 	def dual_domain_constraints_pogs(self, size, voxel_weights=None,
#									 nu_offset=None, nonnegative=False):
# 		raise NotImplementedError

#	def dual_fused_expr_constraints_pogs(self, structure, nu_offset=None,
#										 nonnegative=False):
#			raise NotImplementedError