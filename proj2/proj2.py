#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy Lundberg
#  

from  __future__  import division
from  scipy       import *
from  scipy import linalg as lg
import scipy.optimize as opt
from  matplotlib.pyplot import *
import sys
from numpy.linalg import cholesky, inv, norm, LinAlgError

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
    def gradient(self, x):
        grad = zeros(self.dim)
        h    = 1e-5 
        for i in range(self.dim):
            step    = zeros(self.dim)
            step[i] = h
            grad[i] = (self.f(x+step) - self.f(x-step))/(2.*h)
        return grad

    """ Approximates the hessian using (central) finite differences.
    A symmtrizing step: hessian = .5*(hessian + hessian^T) is
    also performed
    """
    def hessian(self, x):
        hess = zeros((self.dim, self.dim))
        h    = 1e-5
        # Approximates hessian using gradient, see 
        # http://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
        # TODO: We don't need to compute this many values since its
        # symmetric. If we do t more efficiently we don't need
        # the symmetrizing step (I think). - B
        for i in range(self.dim):
            for j in range(self.dim):
                step1     = zeros(self.dim)
                step2     = zeros(self.dim)
                step1[j]  = h
                step2[i]  = h
                grad1 = (self.gradient(x+step1) - self.gradient(x-step1))/(4.*h)
                grad2 = (self.gradient(x+step2) - self.gradient(x-step2))/(4.*h)
                hess[i,j] = grad1[i] + grad2[j]
        # Symmetrizing step. 
        hess = 0.5*(hess + transpose(hess))
        #L = cholesky(hess) # Raises LinAlgError if (but not only if,
                           ## I guess), if hess isn't positive definite.
        return hess

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
            self.gradient = function_gradient
        else:
            self.gradient = FunctionTransforms(objective_function, dimension,
                                            gradient=True)
        self.hessian = FunctionTransforms(objective_function, dimension, 
                                            hessian=True)


class OptimizationMethod(object):
    """
    Super class for various optimization methods
    """
    def __init__(self, opt_problem):
        self.op = opt_problem
         
    def optimize(self):
        pass

class ClassicNewton(OptimizationMethod):
    
    
    def __init__(self, opt_problem):
        super(ClassicNewton, self).__init__(opt_problem)
        

    def optimize(self, guess=None):
        if guess is not None:
            x = guess
        else:
            x = array([0., 0.]) #starting guess
        # x* is a local minimizer if grad(f(x*)) = 0 and 
        # if its hessian is positive definite
        while(True):
            grad = self.op.gradient(x)
            if(norm(grad) < 1e-8):
                return x
            #x = x - dot(inv(self.op.hessian(x)), self.op.gradient(x))

            # use cholesky decomposition as requested in task 3
            # will throw LinAlgError if decomposition fails
            # seems to work at least
            factored = lg.cho_factor(self.op.hessian(x))
            x = x - lg.cho_solve(factored, self.op.gradient(x))

class NewtonExactLine(OptimizationMethod):
    
    
    def __init__(self, opt_problem):
        super(NewtonExactLine, self).__init__(opt_problem)
        

    def optimize(self, guess=None):
        if guess is not None:
            x = guess
        else:
            x = array([0., 0.]) #starting guess
        # x* is a local minimizer if grad(f(x*)) = 0 and 
        # if its hessian is positive definite
        while(True):
            grad = self.op.gradient(x)
            if(norm(grad) < 1e-8):
                return x
            factored = lg.cho_factor(self.op.hessian(x))
            direction = lg.cho_solve(factored, self.op.gradient(x))

            # find step size alpha
            # requires scipy > 0.11 and I have 0.10 // Tobias
            #result = opt.minimize_scalar(lambda alpha: self.op.of(x-alpha*direction),
                                 #bounds=(0, 10))
            alpha = opt.fminbound(
                lambda alpha: self.op.of(x - alpha*direction),
                0, 10)
            x = x - alpha*direction

            

def main():
    def rosenbrock(x):
        return 100*(x[1]-x[0])**2+(1-x[0])**2
    def rosenbrock_grad(x):
        return array([-200*(x[1]-x[0]) -2*(1-x[0]),
                        200*(x[1]-x[0]) ])
    def F(x):
        return x[0]**2 + x[0]*x[1] + x[1]**2
    def F_grad(x):
        return array([2*x[0]+x[1],x[0]+2*x[1]])
    opt = OptimizationProblem(rosenbrock, 2)
    cn  = ClassicNewton(opt)
    print cn.optimize([-3, 7])
    cn  = NewtonExactLine(opt)
    print cn.optimize([-3, 7])


if __name__ == '__main__':
    main()
