# from numpy import zeros, ndarray
# from scipy.sparse import csr_matrix, csc_matrix

# from conrad.compat import *
# from conrad.defs import vec, is_vector, sparse_or_dense

# class DataMatrix(object):
# 	def __init__(self, *args, **kwargs):
# 		A = self.parse_args(*args, **kwargs)
# 		if A is None:
# 			A = self.parse_string_args(*args, **kwargs)


# 	@staticmethod
# 	def parse_dictionary(self, dictionary):
# 		pass

# 	@staticmethod
# 	def parse_yaml(self, yaml_filename):
# 		pass

# 	@staticmethod
# 	def parse_json(self, json_filename):
# 		pass

# 	@staticmethod
# 	def parse_string_args(self, *args, **kwargs):
# 		pass

# 	@staticmethod
# 	def parse_args(self, *args, **kwargs):
# 		arglist = []
# 		for a in list(args) + list(kwargs.values()):
# 			if isinstance(a, list):
# 				arglist += a
# 			if isinstance(a, dict):
# 				arglist += list(a.values())
# 			else:
# 				arglist += a


# 		for a in arglist:
# 			if sparse_or_dense(a):
# 				return A

# 		CSR = 'csr' in arglist or 'CSR' in arglist
# 		CSC = 'csc' in arglist or 'CSC' in arglist
# 		if not (CSR or CSC):
# 			CSC |=
# 			CSR |=
# 		if CSR and CSC:
# 			raise ValueError('sparse matrix format cannot be CSR and CSC')



# 		if 'nzval' in kwargs:
# 		elif 'val' in kwargs:

# 		if 'rowval' in kwargs:
# 		elif 'rowind' in kwargs:
# 		elif 'ind' in kwargs:
