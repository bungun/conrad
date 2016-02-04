from setuptools import setup

setup(
    name='conrad',
    version='0.0.1a',
    author='Baris Ungun, Anqi Fu, Stephen Boyd',
    author_email='ungun@stanford.edu',
    url='http://github.com/bungun/conrad/',
    package_dir={'conrad': 'conrad'},
    packages=[],
    license='GPLv3',
    zip_safe=False,
    description='A convex optimization framework for radiation therapy treament planning',
    long_description=LONG_DESC,
    install_requires=['cvxpy',
                      'numpy >= 1.8',
                      'toolz"'],
    test_suite = 'nose.collector',
    tests_require = ['nose']
    use_2to3=True,
)
