__all__ = ["prox_rank1_hinge"]

from .prox_rank1_generic import prox_rank1_generic
import numpy as np

def prox_rank1_hinge(*args, **kwargs):
	"""
	PROX_RANK1_HINGE returns the scaled proximity operator for the hinge loss

	x = prox_rank1_hinge( x0, D, u )
			where 
			x = argmin_{x} h(x) + 1/2||x-x0||^2_{V}
			and
			V^{-1} = D + u*u'  (or diag(D) + u*u' if D is a vector)
			"D" must be diagonal and positive. "u" can be any vector.

	Here, h(x) = sum(max(0,1-x)), a.k.a., the hinge-loss

	There are also variants:
	x = prox_rank1_hinge( x0, D, u, lambda, linTerm, sigma, inverse)
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
	
	prox = lambda x, t : (np.abs(x)-t).clip(0)
	prox_brk_pts = lambda s : np.hstack((np.ones(s.shape),1.0-s))
		
	return prox_rank1_generic(prox, prox_brk_pts, *args, **kwargs)