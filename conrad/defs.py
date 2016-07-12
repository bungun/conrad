from os import getenv
from conrad.compat import *

def println(*args):
	print(args)

CONRAD_DEBUG = getenv('CONRAD_DEBUG', False)
CONRAD_DEBUG_PRINT = println if CONRAD_DEBUG else lambda x : None

SOLVER_OPTIONS = ['ECOS', 'SCS']
MAX_VERBOSITY = 1
