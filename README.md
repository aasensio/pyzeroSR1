# pyzeroSR1
The zeroSR1 toolbox implements the algorithm from 'A quasi-Newton proximal 
splitting method' by Stephen Becker and Jalal Fadili, which appeared in NIPS 2012. The paper is available at arXiv 1206.1156.

This is a translation to Python of the toolbox found in https://github.com/stephenbeckr/zeroSR1

## Dependencies
This code only depends on `numpy`

## Main usage
Example of usage

	import pyzeroSR1
	import scipy.linalg
	import numpy as np

	N = 12
	tmp = scipy.linalg.hilbert(N)
	A = lambda x : tmp.dot(x)
	At = lambda x : tmp.T.dot(x)
	b = np.ones((N,1))
	l = 1e-1
			
	Q   = At(A(np.identity(N)))    
	c   = At(b)
	normQ = np.linalg.norm(Q)    # Lipschitz constant

	prox = lambda x0, d, u, varargin = None : pyzeroSR1.proxes.prox_rank1_l1(x0, d, u, l)
	h = lambda x : l * np.linalg.norm(x,1)
	fcnGrad = lambda x : pyzeroSR1.smoothFunctions.normSquared(x, A, At, b)

	opts = {'tol': 1e-6, 'grad_tol' : 1e-6, 'nmax' : 1000, 'verbose' : False, 'N' : N, 'L': normQ, 'verbose': 25}

	xk, nIteration, stepSizes = pyzeroSR1.zeroSR1(fcnGrad, h,prox, opts)
