"""
Copyright 2016 Baris Ungun, Anqi Fu

This file is part of CONRAD.

CONRAD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CONRAD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CONRAD.  If not, see <http://www.gnu.org/licenses/>.
"""
from setuptools import setup

LONG_DESC = "TODO: LONG DESCRIPTION"

setup(
    name='conrad',
    version='0.0.2',
    author='Baris Ungun, Anqi Fu, Stephen Boyd',
    author_email='ungun@stanford.edu',
    url='http://github.com/bungun/conrad/',
    package_dir={'conrad': 'conrad'},
    packages=['conrad',
              'conrad.abstract',
              'conrad.medicine',
              'conrad.physics',
              'conrad.optimization',
              'conrad.io',
              'conrad.io.accessors',
              'conrad.visualization',
              'conrad.visualization.plot',
              'conrad.tests'],
    license='GPLv3',
    zip_safe=False,
    description='A convex optimization framework for radiation therapy treament planning',
    long_description=LONG_DESC,
    install_requires=['cvxpy',
                      'numpy >= 1.9',
		                  'scipy >= 0.15',
                      'pyyaml',
                      'nose',
                      'six'],
    test_suite='nose.collector',
    tests_require=['nose'],
    use_2to3=True,
)