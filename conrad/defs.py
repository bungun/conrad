from os import getenv

def println(*args):
	print(args)

def assertln(test):
	assert(test)

CONRAD_DEBUG = getenv('CONRAD_DEBUG', False)
CONRAD_DEBUG_PRINT = println if CONRAD_DEBUG else lambda x : None
CONRAD_DEBUG_ASSERT = assertln if CONRAD_DEBUG else lambda x : None