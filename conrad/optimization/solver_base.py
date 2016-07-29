from conrad.compat import *

GAMMA_DEFAULT = 1e-2
RELTOL_DEFAULT = 1e-3
ABSTOL_DEFAULT = 1e-4
VERBOSE_DEFAULT = 1
MAXITER_DEFAULT = 2000
INDIRECT_DEFAULT = False
GPU_DEFAULT = False

PRIORITY_1 = 9
PRIORITY_2 = 4
PRIORITY_3 = 1

class Solver(object):
	def __init__(self):
		self.use_2pass = False
		self.use_slack = True
		self.__x = None
		self.__gamma = GAMMA_DEFAULT
		self.dvh_vars = {}
		self.slack_vars = {}
		self.feasible = False

	@property
	def gamma(self):
		return self.__gamma

	@gamma.setter
	def gamma(self, gamma):
		if gamma:
			self.__gamma = float(gamma)

	@staticmethod
	def get_cd_from_wts(wt_over, wt_under):
		""" TODO: docstring """
		c = (wt_over + wt_under) / 2.
		d = (wt_over - wt_under) / 2.
		return c, d

	def gamma_prioritized(self, priority):
		priority = int(priority)
		if priority == 1:
			return self.gamma * PRIORITY_1
		elif priority == 2:
			return self.gamma * PRIORITY_2
		elif priority == 3:
			return self.gamma * PRIORITY_3
		elif priority == 0:
			raise ValueError('priority 0 constraints should not have '
							 'slack variables or associated slack '
							 'penalties (gamma)')
		else:
			raise ValueError('argument "priority" must be one of: '
							 '{1, 2, 3}')

	def init_problem(self, n_beams, **options):
		raise RuntimeError('solver method "init_problem" not implemented')

	def clear(self):
		raise RuntimeError('solver method "clear" not implemented')

		def __check_dimensions(self, structures):
			columns = [s.A.shape[1] for s in structures]
			if not all([col == self.n_beams for col in columns]):
				raise ValueError('all structures in plan must have full dose '
								 'matrices with # columns that match # beams in '
								 'the plan. \n # beams: {}\n provided matrix '
								 'shapes: {}'.format(n_beams,
								 [(s.name, s.A.shape) for s in structures]))
			columns = [s.A_mean.size for s in structures]
			if not all([col == self.n_beams for col in columns]):
				raise ValueError('all structures in plan must have mean dose '
								 'vectors with # columns that match # beams in the'
								 ' plan. \n # beams: {}\n provided matrix shapes: '
								 '{}'.format(n_beams,
								 [(s.name, s.A_mean.sisze) for s in structures]))

	def __gather_matrix_and_coefficients(self, structures):
		self.__check_dimensions(structures)

		rows = sum([s.size if not s.collapsable else 1 for s in structures])
		cols = self.n_beams
		A = zeros((rows, cols))
		dose = zeros(rows)
		weight_abs = zeros(rows)
		weight_lin = zeros(rows)
		ptr = 0

		for s in structures:
			if s.collapsable:
				A[ptr, :] = s.A_mean[:]
				weight_abs[ptr] = s.w_over * sum(s.voxel_weights)
				weight_lin[ptr] = 0
				ptr += 1
			else:
				A[ptr : ptr + s.size, :] += s.A_full
				if s.is_target:
					c_, d_ = self.get_cd_from_wts(s.w_over, s.w_under)
					dose[ptr : ptr + s.size] = s.dose
					weight_abs[ptr : ptr + s.size] = c_ * s.voxel_weights
					weight_lin[ptr : ptr + s.size] = d_ * s.voxel_weights
				else:
					dose[ptr : ptr + s.size] = 0
					weight_abs[ptr : ptr + s.size] = s.w_over * s.voxel_weights
					weight_lin[ptr : ptr + s.size] = 0
				ptr += s.size

		return A, dose, weight_abs, weight_lin

	def __construction_report(self, structures):
		report = []
		for structure in structures:
			A = structure.A
			matrix_info = str('using dose matrix, dimensions {}x{}'.format(
							  *structure.A.shape))
			if structure.is_target:
				reason  = 'structure is target'
			else:
				if structure.collapsable:
					A = structure.A_mean
					matrix_info = str('using mean dose, dimensions '
									  '1x{}'.format(structure.A_mean.size))
					reason = str('structure does NOT have '
								 'min/max/percentile dose constraints')
				else:
					reason = str('structure has min/max/percentile '
								 'dose constraints')

			report.append(str('structure {} (label = {}): '
							  '{} (reason: {})'.format(structure.name,
							  structure.label, matrix_info, reason)))
		return report

	def build(self, structures, exact=False):
		raise RuntimeError('solver method "build" not implemented')

	def get_slack_value(Self, constraint_id):
		raise RuntimeError('solver method "get_slack_value" not implemented')

	def get_dvh_slope(self, constraint_id):
		raise RuntimeError('solver method "get_dvh_slope" not implemented')