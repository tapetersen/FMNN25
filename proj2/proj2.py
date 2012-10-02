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
    
    
    def __init__(self, function, gradient = False, hessian = False):
        if( not (gradient or hessian)):
            raise Exception("you must specify a transform")
        self.grad = gradient
        self.hess = hessian
        self.f    = function

    def gradient(self,x):
        pass

    def hessian(self,x):
        pass


    def __call__(self, x):
        if(self.grad):
            return self.gradient(x)
        if(self.hess):
            return self.hessian(x)
        raise Exception("Transform incompletely specified")

class OptimizationProblem(object):
    
    def __init__(self, objective_function, function_gradient = None):
        self.of = objective_function
        if(function_gradient is not None):
            self.fg = function_gradient
        else:
            self.fg = FunctionTransforms(objective_function,gradient=True)
        self.hs = FunctionTransforms(objective_function,hessian=True)

class OptimizationMethod(object):
    """
    Super class for various optimization methods
    """
     def __init__(self,opt_problem):
         self.op = opt_problem
         


class ClassicNewton(OptimizationMethod):
    
    
    def __init__(self,opt_problem):
        super(ClassicNewton,self).__init__(opt_problem)
        



 


def main():
	pass


if __name__ == '__main__':
    main()
