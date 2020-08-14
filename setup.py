import os
from numpy import get_include
from setuptools import setup, Extension
from Cython.Build import cythonize

include_dirs = [get_include()]

ext_modules = [
    Extension("fast_cd.svm", sources=["fast_cd/svm.pyx"], include_dirs=include_dirs),
    Extension("fast_cd.elasticnet", sources=["fast_cd/elasticnet.pyx"], include_dirs=include_dirs)
]

setup(
    name="CoLA",
    version="1.0.0",
    install_requires=[
        "Cython",
        "mpi4py",
        "numpy",
        "sklearn",
        "pandas",
        "joblib",
        "matplotlib",
        "scipy",
        "click"
    ],
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': 3}),
    packages=[
        'cola',
        'fast_cd',
    ],
    entry_points={
        'console_scripts': [
            'colatools=scripts.colatools:cli',
            'run-cola=scripts.run_cola:main',
        ]
    },
    scripts=[]
)
