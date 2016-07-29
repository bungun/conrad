from os import getenv
from pip import get_installed_distributions
from numpy import nan, squeeze, array, ndarray
from scipy.sparse import csr_matrix, csc_matrix
from conrad.compat import *

def println(*args):
	print(args)

CONRAD_DEBUG = getenv('CONRAD_DEBUG', False)
CONRAD_DEBUG_PRINT = println if CONRAD_DEBUG else lambda x : None

SOLVER_OPTIONS = ['ECOS', 'SCS']
MAX_VERBOSITY = 1

CONRAD_MATRIX_TYPES = (ndarray, csr_matrix, csc_matrix)

def vec(vectorlike):
	return squeeze(array(vectorlike))

def is_vector(vectorlike):
	if isinstance(vectorlike, ndarray):
		return len(vectorlike) == vectorlike.shape[0]
	return False

def sparse_or_dense(matrixlike):
	if matrixlike is None:
		return False

	dense = isinstance(matrixlike, ndarray)
	if dense:
		dense &= len(matrixlike.shape) == 2
	sparse = isinstance(matrixlike, (csr_matrix, csc_matrix))
	return sparse or dense

def positive_real_valued(val):
	if val is not None and val is not nan and isinstance(val, (int, float)):
		if val > 0:
			return True
	else:
		return False

def module_installed(name, version_string=None):
	modules = get_installed_distributions()
	index = -1

	installed = False
	for idx, module in enumerate(modules):
		if name in str(module):
			index = idx
			installed = True

	if version_string:
		installed &= str(version_string) in str(modules[index])

	return installed