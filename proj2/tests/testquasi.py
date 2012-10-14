#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy Lundberg
#  
from  __future__  import division

import scipy.optimize as so
import sys

from scipy       import linspace, array, double, all
from scipy import linalg as lg
from matplotlib.pyplot import *
from scipy.optimize import rosen, rosen_der, rosen_hess
from numpy.linalg import cholesky, inv, norm, LinAlgError

if __name__ == "__main__" and __package__ is None:
            __package__ = "tests"

from .. import chebyquad as c
from .. import proj2 as p


"""
Tests the solvers on the chebyquad problem in 4, 8 dimensions.
"""


near = lambda x,y: abs(c.chebyquad(x)-c.chebyquad(y)) < 1e-8
""" Helper method for solver tests"""
    
sol4 = (so.fmin_bfgs(c.chebyquad, linspace(0, 1, 4), c.gradchebyquad))
sol8 = (so.fmin_bfgs(c.chebyquad, linspace(0, 1, 8), c.gradchebyquad))
sol11 = (so.fmin_bfgs(c.chebyquad, linspace(0, 1, 11), c.gradchebyquad))

def test_quasi_newton_broyden():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonBroyden(op)
    assert near(sol4,(cn.optimize(guess)))
    guess=linspace(0,1,8)
    assert near(sol8,(cn.optimize(guess)))
    guess=linspace(0,1,11)
    assert near(sol11,(cn.optimize(guess)))
    
def test_quasi_newton_broyden_bad():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonBroydenBad(op)
    assert near(sol4,(cn.optimize(guess)))
    guess=linspace(0,1,8)
    assert near(sol8,(cn.optimize(guess)))

    #guess=linspace(0,1,11)
    #assert near(sol11,(cn.optimize(guess)))
    
    
def test_quasi_newton_BFSG():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonBFSG(op)
    assert near(sol4,(cn.optimize(guess)))
    
    guess=linspace(0,1,8)
    k = (cn.optimize(guess))
    assert near(sol8,k)

    guess=linspace(0,1,11)
    assert near(sol11,(cn.optimize(guess)))
    
    
def test_quasi_newton_DFP():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonDFP(op)
    assert near(sol4,(cn.optimize(guess)))
    
    guess=linspace(0,1,8)
    k = (cn.optimize(guess))
    print norm(k - sol8)
    assert near(sol8,k)

    guess=linspace(0,1,11)
    assert near(sol11,(cn.optimize(guess)))
    
