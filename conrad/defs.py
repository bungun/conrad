from os import getenv
from numpy import nan, squeeze, array, ndarray
from scipy.sparse import csr_matrix, csc_matrix
from conrad.compat import *

def println(*args):
	print(args)

CONRAD_DEBUG = getenv('CONRAD_DEBUG', False)
CONRAD_DEBUG_PRINT = println if CONRAD_DEBUG else lambda x : None

SOLVER_OPTIONS = ['ECOS', 'SCS']
MAX_VERBOSITY = 1

def vec(vectorlike):
	return squeeze(array(vectorlike))

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