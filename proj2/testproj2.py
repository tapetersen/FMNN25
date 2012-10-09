#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy Lundberg
#  

from  __future__  import division
from  scipy       import *
from  scipy import linalg as lg
from  matplotlib.pyplot import *
import sys
from numpy.linalg import cholesky, inv, norm, LinAlgError
import proj2 as p

def test_hessian():
    """ Test the the numerical hessian for the rosenbrock function 
    and a function of a higher order
    """
    def rosenbrock(x):
        return 100*(x[1]-x[0])**2+(1-x[0])**2
    def rosenbrock_hessian(x):
        return array([ [202, -200] , 
                        [-200,  200]])
    def F(x):
        return x[0]**3+x[1]**3+x[2]**3+x[0]**2 *x[1]**2 *x[2]**2
    def F_hessian(x):
        return array([ [6.*x[0] + 2.*x[1]**2*x[2]**2, 4.*x[0]*x[1]*x[2]**2, 4.*x[0]*x[1]**2 *x[2]] , 
                        [4.*x[0]*x[1]*x[2]**2, 6.*x[1] + 2.*x[0]**2*x[2]**2, 4.*x[0]**2 *x[1]*x[2]] ,
                        [4.*x[0]*x[1]**2 *x[2], 4.*x[0]**2 *x[1]*x[2],6.*x[2] + 2.*x[0]**2*x[1]**2 ]])
    opt1 = p.OptimizationProblem(rosenbrock,2)
    opt2 = p.OptimizationProblem(F,3)
    for i in range(-3,3):
        for j in range(-3,3):
            k  = opt1.hessian([float(i),float(j)]) - \
                rosenbrock_hessian([float(i),float(j)])
            kk  = opt2.hessian([float(i),float(j),float(j+i)]) - \
                F_hessian([float(i),float(j),float(j+i)])
            print k, abs(k) <1e-2
            print kk, abs(kk) <1e-2
            assert(sum( abs(k) <1e-2 )==4 and (sum( abs(kk) <1e-2 )==9) )

def test_solvers_chebyquad():
    """
    Tests the different solvers on the rosenbrock function.
    """
    def rosenbrock(x):
        return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    near = lambda x,y: sum(abs(x-y) < 1e-4) == x.size
        
    sol = array([1.,1.])
    guess = array([-1.,-1.])
    op = p.OptimizationProblem(rosenbrock)
    
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
 

 
def test_solvers_rosenbrock():
    """
    Tests the different solvers on the rosenbrock function.
    """
    def rosenbrock(x):
        return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    near = lambda x,y: sum(abs(x-y) < 1e-4) == x.size
        
    sol = array([1.,1.])
    guess = array([-1.,-1.])
    op = p.OptimizationProblem(rosenbrock)
    
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
    def rosenbrock(x):
        return 100*(x[1]-x[0])**2+(1-x[0])**2
    def rosenbrock_grad(x):
        return array([-200*(x[1]-x[0]) -2*(1-x[0]),
                        200*(x[1]-x[0]) ])
    def F(x):
        return x[0]**2 + x[0]*x[1] + x[1]**2
    def F_grad(x):
        return array([2*x[0]+x[1],x[0]+2*x[1]])
    opt1 = p.OptimizationProblem(rosenbrock)
    opt2 = p.OptimizationProblem(F)
    for i in range(-3,3):
        for j in range(-3,3):
            k  = opt1.gradient([float(i),float(j)]) - \
                rosenbrock_grad([float(i),float(j)])
            kk = opt2.gradient([float(i),float(j)]) - \
                 F_grad([float(i),float(j)])
            assert(sum( abs((k+kk)) <1e-5 )==2)
