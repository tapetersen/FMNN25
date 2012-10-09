#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy Lundberg
#  

from  __future__  import division
from  scipy       import *
from  scipy import linalg as lg
from  matplotlib.pyplot import *
import scipy.optimize as so
import chebyquad as c
import sys
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


near = lambda x,y: sum(abs(x-y) < 1e-4) == x.size
def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    
sol4 = sort(so.fmin_bfgs(c.chebyquad,linspace(0,1,4),c.gradchebyquad))
sol8 = sort(so.fmin_bfgs(c.chebyquad,linspace(0,1,8),c.gradchebyquad))
solr = array([1.,1.])

def test_classic_newton():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.ClassicNewton(op)
    assert near(sol4,sort(cn.optimize(guess)))
    guess=linspace(0,1,8)
    #assert near(sol8,sort(cn.optimize(guess)))
    
    op = p.OptimizationProblem(rosenbrock)
    cn  = p.ClassicNewton(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert near(solr,sort(cn.optimize(guess)))
    
def test_newton_exact_line_search():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.NewtonExactLine(op)
    assert near(sol4,sort(cn.optimize(guess)))
    guess=linspace(0,1,8)
    #assert near(sol8,sort(cn.optimize(guess)))
    
    op = p.OptimizationProblem(rosenbrock)
    cn  = p.NewtonExactLine(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert near(solr,sort(cn.optimize(guess)))
    
def test_newton_inexact_line_search():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.NewtonInexactLine(op)
    assert near(sol4,sort(cn.optimize(guess)))
    guess=linspace(0,1,8)
    #assert near(sol8,sort(cn.optimize(guess)))
    op = p.OptimizationProblem(rosenbrock)
    cn  = p.NewtonInexactLine(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert near(solr,sort(cn.optimize(guess)))

def test_quasi_newton_broyden():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonBroyden(op)
    assert near(sol4,sort(cn.optimize(guess)))
    guess=linspace(0,1,8)
    #assert near(sol8,sort(cn.optimize(guess)))
    
    op = p.OptimizationProblem(rosenbrock)
    cn  = p.QuasiNewtonBroyden(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert near(solr,sort(cn.optimize(guess)))
    
def test_quasi_newton_broyden_bad():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonBroydenBad(op)
    assert near(sol4,sort(cn.optimize(guess)))
    
    guess=linspace(0,1,8)
    
    #assert near(sol8,sort(cn.optimize(guess)))
    
    op = p.OptimizationProblem(rosenbrock)
    cn  = p.QuasiNewtonBroydenBad(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert near(solr,sort(cn.optimize(guess)))
    
def test_quasi_newton_BFSG():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonBFSG(op)
    assert near(sol4,sort(cn.optimize(guess)))
    
    guess=linspace(0,1,8)
    #assert near(sol8,sort(cn.optimize(guess)))
    
    op = p.OptimizationProblem(rosenbrock)
    cn  = p.QuasiNewtonBFSG(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert near(solr,sort(cn.optimize(guess)))
    
def test_quasi_newton_DFP():
    op = p.OptimizationProblem(c.chebyquad)
    guess=linspace(0,1,4)
    cn  = p.QuasiNewtonDFP(op)
    assert near(sol4,sort(cn.optimize(guess)))
    
    guess=linspace(0,1,8)
    
    #assert near(sol8,sort(cn.optimize(guess)))
    
    op = p.OptimizationProblem(rosenbrock)
    cn  = p.QuasiNewtonDFP(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert near(solr,sort(cn.optimize(guess)))


def test_solvers_chebyquad():
    """
    Tests the different solvers on the rosenbrock function.
    """
    near = lambda x,y: norm(x-y) < 1e-2
        
    sol = array([1.,1.])
    guess = array([-1.,-1.])
    op = p.OptimizationProblem(rosen)
    
    print "Testing ClassicNewton"
    cn  = p.ClassicNewton(op)
    assert near(sol,cn.optimize(guess))
    
    print "Testing NewtonInexactLine"
    cn  = p.NewtonInexactLine(op)
    assert near(sol,cn.optimize(guess))
    
    print "Testing NewtonExactLine"
    cn  = p.NewtonExactLine(op)
    assert near(sol,cn.optimize(guess))
    
    #print "Testing QuasiNewtonBroyden"
    #cn  = p.QuasiNewtonBroyden(op)
    #assert near(sol,cn.optimize(guess))
    
    print "Testing QuasiNewtonBroydenBad"
    cn  = p.QuasiNewtonBroydenBad(op)
    assert near(sol,cn.optimize(guess))
    
    print "Testing QuasiNewtonBFSG"
    cn  = p.QuasiNewtonBFSG(op)
    assert near(sol,cn.optimize(guess))
    
    print "Testing QuasiNewtonDFP"
    cn  = p.QuasiNewtonDFP(op)
    print norm(sol-cn.optimize(guess))
    assert near(sol,cn.optimize(guess))
 

 
def test_solvers_rosenbrock():
    """
    Tests the different solvers on the rosenbrock function.
    """
    near = lambda x,y: norm(x-y) < 1e-2
        
    sol = array([1.,1.])
    guess = array([-1.,-1.])
    op = p.OptimizationProblem(rosen)
    
    print "Testing ClassicNewton"
    cn  = p.ClassicNewton(op)
    assert near(sol,cn.optimize(guess))
    
    print "Testing NewtonInexactLine"
    cn  = p.NewtonInexactLine(op)
    assert near(sol,cn.optimize(guess))
    
    print "Testing NewtonExactLine"
    cn  = p.NewtonExactLine(op)
    assert near(sol,cn.optimize(guess))
    
    #print "Testing QuasiNewtonBroyden"
    #cn  = p.QuasiNewtonBroyden(op)
    #assert near(sol,cn.optimize(guess))
    
    print "Testing QuasiNewtonBroydenBad"
    cn  = p.QuasiNewtonBroydenBad(op)
    assert near(sol,cn.optimize(guess))
    
    print "Testing QuasiNewtonBFSG"
    cn  = p.QuasiNewtonBFSG(op)
    assert near(sol,cn.optimize(guess))
    
    print "Testing QuasiNewtonDFP"
    cn  = p.QuasiNewtonDFP(op)
    assert near(sol,cn.optimize(guess))
    

def test_minimize():
    """ Test the cubic_minimize help function
    """
    near = lambda a, b: abs(a-b) < 1e-3

    # basic minimum
    assert near(.5, p.cubic_minimize(1., -1, 1, 1, 0, 1))

    # scaling and translation of interval
    assert near(1, p.cubic_minimize(1., -1, 1, 1, 0, 2))
    assert near(2, p.cubic_minimize(1., -1, 1, 1, 1, 3))

    # endpoints for concave function
    assert near(0, p.cubic_minimize(1.0, 1, 1.1, -1, 0, 1))
    assert near(1, p.cubic_minimize(1.1, 1, 1.0, -1, 0, 1))

    # non trivial minimum
    assert near(1/3, p.cubic_minimize(1., -1, 1, 0, 0, 1))

    #scaled
    assert near((1/3)*2+3, p.cubic_minimize(1., -1, 1, 0, 3, 5))
    



def test_gradient():
    """ Test the gradient using rosenbrock and higher order functions
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
