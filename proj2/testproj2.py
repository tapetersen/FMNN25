#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy Lundberg
#  
from  __future__  import division

import scipy.optimize as so
import chebyquad as c
import sys

from  scipy       import linspace, array, double
from  scipy import linalg as lg
from  matplotlib.pyplot import *
from scipy.optimize import rosen, rosen_der, rosen_hess
from numpy.linalg import cholesky, inv, norm, LinAlgError
import proj2 as p

def test_hessian():
    """ Test the the numerical hessian for the rosenbrock function 
    with and without explicit gradient and a function of a higher order
    """
    def F(x):
        return x[0]**3+x[1]**3+x[2]**3+x[0]**2 *x[1]**2 *x[2]**2
    def F_hessian(x):
        return array([ [6.*x[0] + 2.*x[1]**2*x[2]**2, 4.*x[0]*x[1]*x[2]**2, 4.*x[0]*x[1]**2 *x[2]] , 
                        [4.*x[0]*x[1]*x[2]**2, 6.*x[1] + 2.*x[0]**2*x[2]**2, 4.*x[0]**2 *x[1]*x[2]] ,
                        [4.*x[0]*x[1]**2 *x[2], 4.*x[0]**2 *x[1]*x[2],6.*x[2] + 2.*x[0]**2*x[1]**2 ]])
    opt1 = p.OptimizationProblem(rosen)
    opt2 = p.OptimizationProblem(rosen, rosen_der)
    opt3 = p.OptimizationProblem(F,3)
    for i in range(-3,3):
        for j in range(-3,3):
            x = array([i, j, 3], dtype=double)
            k  = opt1.hessian(x) - rosen_hess(x)
            kk  = opt2.hessian(x) - rosen_hess(x)
            kkk  = opt3.hessian(x) - F_hessian(x)
            print k, abs(k) <1e-2
            print kk, abs(kk) <1e-2
            print kkk, abs(kkk) <1e-2
            assert all( abs(k) <1e-2 )
            assert all( abs(kk) <1e-2 )
            assert all( abs(kkk) <1e-2 )


near = lambda x,y: abs(c.chebyquad(x)-c.chebyquad(y)) < 1e-8
""" Helper method for solver tests"""
    
sol4 = (so.fmin_bfgs(c.chebyquad,linspace(0,1,4),c.gradchebyquad))
sol8 = (so.fmin_bfgs(c.chebyquad,linspace(0,1,8),c.gradchebyquad))
solr = array([1.,1.])

def test_classic_newton():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.ClassicNewton(op)
    assert near(sol4,(cn.optimize(guess)))
    guess=linspace(0,1,8)
    assert near(sol8,(cn.optimize(guess)))
    
    op = p.OptimizationProblem(rosen)
    cn  = p.ClassicNewton(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)
    
def test_newton_exact_line_search():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.NewtonExactLine(op)
    assert near(sol4,(cn.optimize(guess)))
    guess=linspace(0,1,8)
    assert near(sol8,(cn.optimize(guess)))
    
    op = p.OptimizationProblem(rosen)
    cn  = p.NewtonExactLine(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)
    
def test_newton_inexact_line_search():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.NewtonInexactLine(op)
    assert near(sol4,(cn.optimize(guess)))
    guess=linspace(0,1,8)
    assert near(sol8,(cn.optimize(guess)))
    op = p.OptimizationProblem(rosen)
    cn  = p.NewtonInexactLine(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)

def test_quasi_newton_broyden():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonBroyden(op)
    assert near(sol4,(cn.optimize(guess)))
    guess=linspace(0,1,8)
    assert near(sol8,(cn.optimize(guess)))
    
    op = p.OptimizationProblem(rosen)
    cn  = p.QuasiNewtonBroyden(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)
    
def test_quasi_newton_broyden_bad():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonBroydenBad(op)
    assert near(sol4,(cn.optimize(guess)))
    
    guess=linspace(0,1,8)
    
    assert near(sol8,(cn.optimize(guess)))
    
    op = p.OptimizationProblem(rosen)
    cn  = p.QuasiNewtonBroydenBad(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)
    
def test_quasi_newton_BFSG():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonBFSG(op)
    assert near(sol4,(cn.optimize(guess)))
    
    guess=linspace(0,1,8)
    k = (cn.optimize(guess))
    assert near(sol8,k)
    
    op = p.OptimizationProblem(rosen)
    cn  = p.QuasiNewtonBFSG(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)
    
def test_quasi_newton_DFP():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonDFP(op)
    assert near(sol4,(cn.optimize(guess)))
    
    guess=linspace(0,1,8)
    k = (cn.optimize(guess))
    print norm(k - sol8)
    assert near(sol8,k)
    
    op = p.OptimizationProblem(rosen)
    cn  = p.QuasiNewtonDFP(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)

    

def test_minimize():
    """ Test the cubic_minimize help function
    """
    near = lambda a, b: abs(a-b) < 1e-3

    # basic minimum
    assert near(.5, p.cubic_minimize(1., -1, 1, 1, 0, 1, 0, 1))

    # basic minimum outside range
    assert near(.6, p.cubic_minimize(1., -1, 1, 1, 0, 1, 0.6, 1))

    # scaling and translation of interval
    assert near(1, p.cubic_minimize(1., -1, 1, 1, 0, 2, 0, 2))
    assert near(2, p.cubic_minimize(1., -1, 1, 1, 1, 3, 0, 2))

    # endpoints for concave function
    assert near(0.1, p.cubic_minimize(1.0, 1, 1.1, -1, 0, 1, 0.1, 0.9))
    assert near(0.9, p.cubic_minimize(1.1, 1, 1.0, -1, 0, 1, 0.1, 0.9))

    # non trivial minimum
    assert near(1/3, p.cubic_minimize(1., -1, 1, 0, 0, 1, 0, 1))
    assert near(2/3, p.cubic_minimize(1., 0, 1, 1, 0, 1, 0, 1))

    # turned around
    assert near(1/3, p.cubic_minimize(1., 0, 1, -1, 1, 0, 0, 1))
    assert near(2/3, p.cubic_minimize(1., 1, 1, 0, 1, 0, 0, 1))

    #scaled
    assert near((1/3)*2+3, p.cubic_minimize(1., -1, 1, 0, 3, 5, 3, 5))
    

def test_linesearch():
    """ Tests the linesearch on example in book """

    f = lambda alpha: 100*alpha**4+(1-alpha)**2
    f_grad = lambda alpha, _=None: 400*alpha**3-2*(1-alpha)

    alpha = p.find_step_size(f, f_grad, debug=True)
    print alpha
    assert 0.15 < alpha and alpha < 0.17


def test_gradient():
    """ Test the gradient using rosen and higher order functions
    """
    def F(x):
        return x[0]**2 + x[0]*x[1] + x[1]**2
    def F_grad(x):
        return array([2*x[0]+x[1],x[0]+2*x[1]])
    opt1 = p.OptimizationProblem(rosen)
    opt2 = p.OptimizationProblem(F)
    for i in range(-3,3):
        for j in range(-3,3):
            k  = opt1.gradient([float(i),float(j)]) - \
                rosen_der([float(i),float(j)])
            kk = opt2.gradient([float(i),float(j)]) - \
                 F_grad([float(i),float(j)])
            assert(sum( abs((k+kk)) <1e-5 )==2)
