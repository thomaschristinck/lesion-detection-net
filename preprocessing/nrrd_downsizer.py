import numpy as np
import nrrd
import os
from os import listdir
from os.path import join

files = sorted(listdir('../images'))

for f in files:
	if f.endswith('netseg.nrrd'):
		netseg, opts = nrrd.read(f)
		netseg = netseg.astype(np.float32)
		_OPTS = {'space': 'RAS', 'space directions': [(1, 0, 0), (0, 1, 0), (0, 0, 3)]}
		nrrd.write(join('../images', f), netseg, options=_OPTS)
	if f.endswith('uncmcvar.nrrd'):
		uncmcvar, opts = nrrd.read(f)
		uncmcvar = uncmcvar.astype(np.float32)
		_OPTS = {'space': 'RAS', 'space directions': [(1, 0, 0), (0, 1, 0), (0, 0, 3)]}
		nrrd.write(join('../images', f), uncmcvar, options=_OPTS)
