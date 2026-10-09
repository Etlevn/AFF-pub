from setuptools import setup, find_packages
import os
from setuptools import dist

dist.Distribution().fetch_build_eggs(['Cython', 'numpy'])

import numpy
from Cython.Build import cythonize

required = [
    "pytest",
    "cython",
    "numpy>=1.21.0",
    "tensorflow>=2.15.0",
    "sympy",
    "pandas",
    "scikit-learn",
    "click",
    "deap",
    "pathos",
    "seaborn",
    "tqdm",
    "commentjson",
    "PyYAML",
    "prettytable"
]

extras = {
    "control": [
        "mpi4py",
        "gym[box2d]==0.15.4",
        "pybullet",
        "stable-baselines[mpi]==2.10.0"
    ],
    "regression": []
}
extras['all'] = list(set([item for group in extras.values() for item in group]))

setup(  name='dso',
        version='1.0dev',
        description='Deep symbolic optimization.',
        author='LLNL',
        packages=find_packages(),
        setup_requires=["numpy", "Cython"],
        ext_modules=cythonize(['cyfunc.pyx']),
        include_dirs=[numpy.get_include()],
        install_requires=required,
        extras_require=extras
        )
