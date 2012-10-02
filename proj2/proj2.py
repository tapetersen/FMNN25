#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy Lundberg
#  

from  __future__  import division
from  scipy       import *
from  scipy import linalg as lg
from  matplotlib.pyplot import *
import sys

class FunctionTransforms(object):
    """ A class which provides a transform of a given function. 
    Provided transforms: 
        - gradient
        - hessian
    Usage: initialize an instance of the class where the function
    is provided and the type of transform is provided (set gradient
    or hessian to true). The call method then returns the choosen
    transfrom
    The transforms are constructed by finite differences. 
    """
    
    
    def __init__(self, function, dimension,
                 gradient = False, hessian = False):
        if( not (gradient or hessian)):
            raise Exception("you must specify a transform")
        self.grad = gradient
        self.hess = hessian
        self.f    = function
        self.dim  = dimension

    """Approximates the gradient using (central) finite differences 
    of degree 1
    """
    def gradient(self,x):
        grad = zeros(self.dim)
        h    = 1e-5 
        for i in range(self.dim)
            grad[i] = (self.f(x+h*.5) - self.f(x-h*.5))/h
        return grad

    """ Approximates the hessian using (central) finite differences 
    of degree 1. 
    A symmtrizing step: hessian = .5*(hessian + hessian^T) is
    also performed
    """
    def hessian(self,x):
        pass


    def __call__(self, x):
        if(self.grad):
            return self.gradient(x)
        if(self.hess):
            return self.hessian(x)
        raise Exception("Transform incompletely specified")

class OptimizationProblem(object):
    """ Provides an interface to various methods on a given function
    which can be used to optimize it. 
    """
    
    """Provide a function, the functions dimension (\in R^n) and
    optionally its gradient (given as a callable function)
    """
    def __init__(self, objective_function, dimension,
                            function_gradient = None):
        self.dim = dimension
        self.of = objective_function
        if(function_gradient is not None):
            self.fg = function_gradient
        else:
            self.fg = FunctionTransforms(objective_function, dimension,
                                            gradient=True)
        self.hs = FunctionTransforms(objective_function, dimension, 
                                            hessian=True)

class OptimizationMethod(object):
    """
    Super class for various optimization methods
    """
    def __init__(self,opt_problem):
        self.op = opt_problem
         
    def optimize(self):
        pass

class ClassicNewton(OptimizationMethod):
    
    
    def __init__(self,opt_problem):
        super(ClassicNewton,self).__init__(opt_problem)
        

    def optimize(self):
        return 'Optimizing with ClassicNewton'

 


def main():
    def rosenbrock(x):
        return 100*(x[1]-x[0])**2+(1-x[0])**2
    opt = OptimizationProblem(rosenbrock,2)
    cn  = ClassicNewton(opt)
    print cn.optimize()


if __name__ == '__main__':
    main()
