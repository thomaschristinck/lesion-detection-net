from setuptools import setup, Extension
import numpy as np

# Compile and install locally. Run 'python setup.py build_ext --inplace'

ext_modules = [
	Extension(
		'tools.mask',
		sources=['tools/maskApi.c', 'tools/mask.pyx'],
		include_dirs = [np.get_include(), 'tools'],
		extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
	)
]

setup(
	name='tools',
	packages='tools',
	package_dir={'tools':'tools'},
	install_requires=[
		'setuptools>=18.0',
		'cython>=0.27.3'
		'matplotlib>=2.1.0'
	],
	version='2.0',
	ext_modules=ext_modules
)