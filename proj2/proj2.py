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
    
    """
    Th constructor receives information abut what kind of transform is needed
    for the moment - only one transform can be specified.

    Exceptions:
        - If no transforms are specified the class asks the user for one
        - If two transforms are specfied the class asks for only one to be specified,
        the transform can only be uniquely specified.
    """
    def __init__(self, function, dimension,
                 gradient = False, hessian = False):
        if( not (gradient or hessian)):
            raise Exception("you must specify a transform")
        elif(gradient and hessian):
            raise Exception("You can only specify one transform");
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

    """ Approximates the hessian using (central) finite differences of degree 2
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
    """
    Evaluation function that performs the transfrm specfied,
    the constructuor ensures that the transform is uniquely determind at instaciation.
    """
    def __call__(self, x):
        if(self.grad):
            return self.gradient(x)
        if(self.hess):
            return self.hessian(x)
        #raise Exception("Transform incompletely specified")
        #This eception is never reached since always one trasform is guaranteed to be specified through the constructor 

class OptimizationProblem(object):
    """ Provides an interface to various methods on a given function
    which can be used to optimize it. 
    """
    
    """The user provides a function, the functions dimension (\in R^n) and
    optionally its gradient (given as a callable function)

    An atribute 'is_function_gradient' is added ss a boolean so that we can keep track
    if we're working with a matrix or a function
    
    """
    def __init__(self, objective_function, dimension,
                            function_gradient = None):
        
        self.dim = dimension
        self.objective_function = objective_function
        self.is_function_gradient = False;
        """
            A gradient is specfied by the user, use it.
            Otherwise - obtain the gradient numerically

            Always construct the Hessian numerically
        """
        if(function_gradient is not None):
            self.gradient = function_gradient
            self.is_function_gradient = True;
        else:
            self.gradient = FunctionTransforms(objective_function, dimension,
                                            gradient=True)
        self.hessian = FunctionTransforms(objective_function, dimension, 
                                            hessian=True)


class OptimizationMethod(object):
    """
    Super class for various optimization methods

    Please note - opt_problem inherits from the class Optimization problem
    """
    def __init__(self, opt_problem):
        self.opt_problem = opt_problem
         
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
            grad = self.opt_problem.gradient(x)
            print norm(grad);
            if(norm(grad) < 1e-5):
                return x
            #x = x - dot(inv(self.opt_problem.hessian(x)), self.opt_problem.gradient(x))

            # use cholesky decomposition as requested in task 3
            # will throw LinAlgError if decomposition fails
            # This is not a problem as if that's the case the point 
            # we're converging to is a saddle point and not a minimum
            try:
                factored = lg.cho_factor(self.opt_problem.hessian(x))
                x = x - lg.cho_solve(factored, self.opt_problem.gradient(x))
            except LinalgError:
                raise LinAlgError(
                    "Hessian not positive definite, converging to saddle point")

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
            grad = self.opt_problem.gradient(x)
            if(norm(grad) < 1e-5):
                return x
            try:
                factored = lg.cho_factor(self.opt_problem.hessian(x))
                direction = lg.cho_solve(factored, self.opt_problem.gradient(x))
            except LinAlgError:
                raise LinAlgError(
                    "Hessian indefinite, converging to saddle point")
            

            # find step size alpha
            # requires scipy > 0.11 and I have 0.10 // Tobias
            #result = opt.minimize_scalar(lambda alpha: self.opt_problem.objective_function(x-alpha*direction),
                                 #bounds=(0, 10))
            alpha = opt.fminbound(
                lambda alpha: self.opt_problem.objective_function(x - alpha*direction),
                0, 1000)
            x = x - alpha*direction


class NewtonInexactLine(OptimizationMethod):    
    
    def __init__(self, opt_problem):
        super(NewtonInexactLine, self).__init__(opt_problem)
        

    def optimize(self, guess=None):
        if guess is not None:
            x = guess
        else:
            x = array([0., 0.]) #starting guess
        # x* is a local minimizer if grad(f(x*)) = 0 and 
        # if its hessian is positive definite
        while(True):
            grad = self.opt_problem.gradient(x)

            #Will only be reached if the user by freak-accident guesses correctly
##            if(norm(grad) < 1e-5):
##                print('FREAK!!!')
##                return x

            try:
                factored = lg.cho_factor(self.opt_problem.hessian(x))
                direction = lg.cho_solve(factored, self.opt_problem.gradient(x))
            except LinAlgError:
                raise LinAlgError(
                    "Hessian indefinite, converging to saddle point")


            # find step size alpha
            # requires scipy > 0.11 and I have 0.10 // Tobias
            #result = opt.minimize_scalar(lambda alpha: self.opt_problem.objective_function(x-alpha*direction),
                                 #bounds=(0, 10))
            alpha = opt.fminbound(
                lambda alpha: self.opt_problem.objective_function(x - alpha*direction),
                0, 1000)
            x = x - alpha*direction
            print x
            print alpha
            #epsilon can take an arbitrary value between 0 and 1,
            #as long as it is fixed.
            epsilon = 0.5

            #Some prework to be used to check for wolfe-conditions
            grad_x = norm( self.opt_problem.gradient(x) )
            grad_step = norm( self.opt_problem.gradient(x - alpha*direction) )
            
            f_x = self.opt_problem.objective_function(x)
            f_step = self.opt_problem.objective_function(x - alpha*direction)
            #for newton-quasi methods the constants in the conditions are:
            c1 = 1e-4
            c2=0.9

            ### THE WOLF CONDITIONS ###

            #i) Armijo rule, first condition in the wolfe conditions
            #   makes sure that the step length alpha is reduced sufficiently 
            if(f_step <= f_x + c1*alpha*norm( direction )*grad_x):

                #ii) Curvature condition to satisfy sufficient slope-reduction,
                #    using the modified strong condition on curvature
                if(norm(direction*grad_step) <= c2*norm(direction*grad_x)):

                    #Everything passed, we're in the right direction
                    #next we make sure the wolfe-goldstein condition is passed,
                    #then we are done!

                    #Precalculated functionvalues used later to check for stop-criteria
                    grad_alpha = norm(self.opt_problem.gradient(alpha))
                    grad_origin = norm(self.opt_problem.gradient(0))
                    f_alpha = norm(self.opt_problem.objective_function(alpha))
                    f_origin = norm(self.opt_problem.objective_function(0))

                    #we check if we've come sufficiently near by the wolf-goldstein condition:
                    #grad(alpha) >= (1-epsilon)*grad(0)
                    if(grad_alpha >= (1-epsilon)*grad_origin):
                        print('GOLDSTEIN!!!')
                        return x
                        #Do not replace 1-epsilon with 0.5, readabillity above effectiveness
                        #for now.
                    else:
                        continue
                else:
                    #else added for readabillity mostly can be removed
                    Exception('Condition (ii) not satisfied in the wolfe-conditions: insufficient decrease in slope')
            else:
                #else added for readabillity mostly can be removed
                Exception('Condition (i) not satisfied in the wolfe-conditions: insufficient reduction in step-size alpha');
                          
                        
            
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
    print "\nClassicNewton.Optimize(...): \n"
    print cn.optimize([-3, -3])
    cn  = NewtonExactLine(opt)
    print "\nNewtonExactLine.Optimize(...): \n"
    print cn.optimize([-3, -3])
    cn = NewtonInexactLine(opt);
    print "\nNewtonInexactLine.Optimize(...): \n"
    print cn.optimize([100, 100])


if __name__ == '__main__':
    main()
