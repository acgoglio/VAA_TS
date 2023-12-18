# -*- coding: utf-8 -*-

#----------------------------------------------------------
#----------------- Tidal Filter ------------------------
#----------------------------------------------------------

import numpy as np
import numpy.fft as F
import scipy as sp
from scipy import signal
import numpy as np

def fft_2bands(nelevation, low_bound, high_bound,
                           low_bound_1, high_bound_1,alpha=0.4,invert='True'):
	""" Performs a pass band filter on the nelevation series.
	low_bound and high_bound specifies the boundaries of the filter.
	"""

	if len(nelevation) % 2:
		result = F.rfft(nelevation, len(nelevation))
	else:
		result = F.rfft(nelevation)

	freq = F.fftfreq(len(nelevation))[:result.shape[0]]

	factor = np.ones_like(result)

	#---- first window --------------
	sl = np.logical_and(high_bound < freq, freq < low_bound)

	#---- second window ------------
	sl_2=np.logical_and(high_bound_1 < freq, freq < low_bound_1)

	a = factor[sl]
	b = factor[sl_2]
	lena = alen = a.shape[0]
	lenb = blen = b.shape[0]

	#---- create a Tukey tapering ---------
	a = 1-sp.signal.tukey(lena,alpha)
	b = 1-sp.signal.tukey(lenb,alpha)

	## Insert Tukey window into factor
	factor[sl] = a[:alen]
	factor[sl_2]=b[:blen]

	if invert=='False':
		result = result * (factor)
	elif invert=='True':
		result = result * (-(factor-1)) # invert the filter

	relevation = F.irfft(result, len(nelevation))
	return relevation,np.abs(factor)
