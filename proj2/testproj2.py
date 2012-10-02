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
    
    

def test_gradient():
    def rosenbrock(x):
        return 100*(x[1]-x[0])**2+(1-x[0])**2
    def rosenbrock_grad(x):
        return array([-200*(x[1]-x[0]) -2*(1-x[0]),
                        200*(x[1]-x[0]) ])
    def F(x):
        return x[0]**2 + x[0]*x[1] + x[1]**2
    def F_grad(x):
        return array([2*x[0]+x[1],x[0]+2*x[1]])
    opt1 = p.OptimizationProblem(rosenbrock,2)
    opt2 = p.OptimizationProblem(F,2)
    for i in range(-3,3):
        for j in range(-3,3):
            k  = opt1.gradient([float(i),float(j)]) - \
                rosenbrock_grad([float(i),float(j)])
            kk = opt2.gradient([float(i),float(j)]) - \
                 F_grad([float(i),float(j)])
            assert(sum( abs((k+kk)) <1e-5 )==2)
