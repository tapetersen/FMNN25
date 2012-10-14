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

from .. import chebyquad as c
from .. import proj2 as p


near = lambda x,y: abs(c.chebyquad(x)-c.chebyquad(y)) < 1e-8
""" Helper method for solver tests"""
    
sol4 = (so.fmin_bfgs(c.chebyquad, linspace(0, 1, 4), c.gradchebyquad))
sol8 = (so.fmin_bfgs(c.chebyquad,linspace(0, 1, 8), c.gradchebyquad))
solr = array([1., 1.])

def test_classic_newton():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.ClassicNewton(op)
    assert near(sol4,(cn.optimize(guess)))
    guess=linspace(0,1,8)
    assert near(sol8,(cn.optimize(guess)))
    
def test_newton_exact_line_search():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.NewtonExactLine(op)
    assert near(sol4,(cn.optimize(guess)))
    guess=linspace(0,1,8)
    assert near(sol8,(cn.optimize(guess)))
    
def test_newton_inexact_line_search():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.NewtonInexactLine(op)
    assert near(sol4,(cn.optimize(guess)))
    guess=linspace(0,1,8)
    assert near(sol8,(cn.optimize(guess)))

