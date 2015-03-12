import numpy as np
import scipy.linalg

N = 12
A = scipy.linalg.hilbert(N)
b = np.ones((N,1))
l = 1e-1
        
Q   = A.T.dot(A)
c   = A.T.dot(b)
normQ = np.linalg.norm(Q)

prox = lambda x0, d, u, varargin : prox_rank1_l1(x0, d, u, l)
h = lambda x : l * np.linalg.norm(x,1)

fcnGrad = lambda x : normSquaredFunction(x, A, b)