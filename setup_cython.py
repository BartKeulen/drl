from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'ddpg cython',
    ext_modules = cythonize('drl/ddpg/ddpg.py'),
)