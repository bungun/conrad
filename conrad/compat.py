from __future__ import print_function
from sys import version_info

if version_info.major > 2:
	xrange = range
	def listmap(f, *args):
		return list(map(f, *args))
	def listfilter(f, *args):
		return list(filter(f, *args))
else:
	listmap = map
	listfilter = filter

