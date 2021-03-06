__all__ = ["prox_rank1_linf"]

from .prox_rank1_generic import prox_rank1_generic
import numpy as np

def prox_rank1_linf(*args, **kwargs):
	"""
	PROJ_RANK1_LINF returns the scaled proximity operator for l_infinity norm constraints

	x = proj_rank1_linf( x0, D, u )
			where 
			x = argmin_{x} h(x) + 1/2||x-x0||^2_{V}
			and
			V^{-1} = D + u*u'  (or diag(D) + u*u' if D is a vector)
			"D" must be diagonal and positive. "u" can be any vector.

	Here, h(x) is the indicator function of the l_infinity ball, i.e.,
		{ x | norm(x,inf) <= 1 }
		To scale the ball, just use the scaling parameter "lambda" (see below)

	There are also variants:
	x = proj_rank1_linf( x0, D, u, lambda, linTerm, sigma, inverse)
			returns
			x = argmin_{x} h(lambda.*x) + 1/2||x-x0||^2_{V} + linTerm'*x
			and
			either V^{-1} = D + sigma*u*u' if "inverse" is true (default)
			or     V      = D + sigma*u*u' if "inverse" is false
			and in both cases, "sigma" is either +1 (default) or -1.
			"lambda" should be non-zero

	Stephen Becker, Feb 26 2014, stephen.beckr@gmail.com
	Reference: "A quasi-Newton proximal splitting method" by S. Becker and J. Fadili
	NIPS 2012, http://arxiv.org/abs/1206.1156

	See also prox_rank1_generic.m	
	
	Python version: A. Asensio Ramos (March 12, 2015)
	"""
	
	prox = lambda x, t : np.sign(x) * np.abs(x).clip(0,1)
	prox_brk_pts = lambda s : np.hstack((-np.ones(s.shape),np.ones(s.shape)))
		
	return prox_rank1_generic(prox, prox_brk_pts, *args, **kwargs)