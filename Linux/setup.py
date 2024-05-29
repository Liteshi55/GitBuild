# setup.py

import numpy as np
from setuptools import setup, Extension,find_packages
from Cython.Build import cythonize

extensions = [
    Extension(name="htrfnwe", sources=["htrfnwe/htrfnwe_v1.pyx"], include_dirs=[np.get_include()]),
]

# Cython compiler directives
compiler_directives = {
    'language_level': "3",
    'boundscheck': False,
    'wraparound': False,
    'initializedcheck': False,
    'cdivision': True,
    'always_allow_keywords': False,
    'unraisable_tracebacks': False,
    'binding': False
}

setup(
    include_dirs=[np.get_include()],
    name="htrfnwe",
    version="0.8",
    description="A package with multiple Cython programs for technical analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Litesh",
    author_email="liteshi55@gmail.com",
    url="https://github.com/Liteshi55/htrfnwe",
    packages=find_packages(include=['htrfnwe']),
    setup_requires=['numpy', 'scikit-learn', 'Cython', 'setuptools','pytest-runner'],
    install_requires=['numpy', 'scikit-learn', 'Cython','pandas'],
    package_dir={'htrfnwe': ''},
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives, annotate=True),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Cython",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
