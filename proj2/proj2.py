#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy Lundberg
#  

from  __future__  import division

#import ipdb
import sys
import scipy.optimize as opt
import scipy.linalg as lg

from scipy       import *
from matplotlib.pyplot import *
from numpy.linalg import cholesky, inv, norm, LinAlgError
from numpy import polynomial as P
from collections import defaultdict

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
    def __call__(self, x):
        """
        Pass through for the objective function
        """
        return self.objective_function(x)


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
        

    def optimize(self, guess=None, debug=False):
        if guess is not None:
            x = guess
        else:
            x = array([0., 0.]) #starting guess

        if debug:
            self.xs = []
        # x* is a local minimizer if grad(f(x*)) = 0 and 
        # if its hessian is positive definite
        
        f        = self.opt_problem.objective_function
        f_grad   = self.opt_problem.gradient
        f_grad_x = self.opt_problem.gradient(x)
        f_x      = self.opt_problem.objective_function(x)
        G = self.opt_problem.hessian(x)
        H = inv(G)
        for ii in xrange(50):
            if debug:
                self.xs.append(x)

            print "x: %f y: %f" % (x[0], x[1])

            if norm(f_grad_x) < 1e-4 :
                break
        
            direction = self.find_direction(f_grad_x, H, G)
            alpha = self.find_step_size(
                f=lambda alpha: f(x - alpha*direction),
                f_grad=lambda alpha: dot(f_grad(x - alpha*direction), -direction))
            
            delta = -alpha*direction
            H, G = self.update_hessian(x, delta, H, G) 
            x = x + delta
            f_grad_x = f_grad(x)

        if debug:
            self.xs = array(self.xs)
            self.plot()

        return x

    def find_step_size(self, f, f_grad):
        return 1

    def find_direction(self, f_grad_x, H, G):
        try:
            factored = lg.cho_factor(G)
            return lg.cho_solve(factored, f_grad_x)
        except LinalgError:
            raise LinAlgError(
                "Hessian not positive definite, converging to saddle point")

    def update_hessian(self, x, delta, H, G):
        return None, self.opt_problem.hessian(x + delta)

    def plot(self):
        # find min/max of points
        xmin, ymin = amin(self.xs, axis=0)
        xmax, ymax = amax(self.xs, axis=0)
        deltax = xmax-xmin
        deltay = ymax-ymin
        xmin = xmin-.05*deltax
        ymin = ymin-.05*deltay
        xmax = xmax+.05*deltax
        ymax = ymax+.05*deltay
        x = linspace(xmin, xmax, 100)
        y = linspace(ymin, ymax, 100)
        X, Y = meshgrid(x, y)
        Z = self.opt_problem.objective_function([X, Y])
        pcolor(X, Y, Z)
        colorbar()
        axis([xmin, xmax, ymin, ymax])

        plot(self.xs[:,0], self.xs[:,1], '+-', color='black')
        show()

        
class NewtonExactLine(ClassicNewton):
    
    
    def __init__(self, opt_problem):
        super(NewtonExactLine, self).__init__(opt_problem)
        
    def find_step_size(self, f, f_grad):
        return opt.fminbound(f, 0, 1000)


class NewtonInexactLine(ClassicNewton):    
    """
    Newton method with inexact line search as given in Fletcher
    """
    
    def __init__(self, *args, **kwargs):
        super(NewtonInexactLine, self).__init__(*args, **kwargs)
        kwargs = defaultdict(lambda: None, **kwargs)
        self.minimum_bound = kwargs['minimum_bound'] or 0.0
        self.rho = kwargs['rho'] or 1e-3
        self.sigma = kwargs['sigma'] or 0.9
        self.tau1 = 9.
        self.tau2 = .1
        self.tau3 = .5
        

    def find_step_size(self, f, f_grad):

        f_0 = f(0)
        f_grad_0 = f_grad(0)
        grad_norm = norm(f_grad_0)

        # Calculate maximum alpha where we would always reject it
        # due to the Armijo rule (condition i)
        mu = (self.minimum_bound - f_0)/(self.rho*f_grad_0)

        # bracketing face, first algorithm part in book

        alpha = min(1., mu*.5)
        alpha_prev = 0.

        f_alpha = f_0
        # Begin the bracketing phase
        while True:
            f_prev_alpha = f_alpha
            f_alpha = f(alpha)

            if f_alpha < self.minimum_bound:
                return alpha

            # check condition 1 or if new f value is higher
            if (f_alpha > f_0 + alpha*self.rho*f_grad_0 or
                    f_alpha >= f_prev_alpha):
                a = alpha_prev
                b = alpha
                f_a = f_prev_alpha
                break

            f_grad_alpha = f_grad(alpha)

            # check condition 2
            if norm(f_grad_alpha) <= -self.sigma*f_grad_0:
                return alpha

            if f_grad_alpha >= 0:
                a = alpha
                b = alpha_prev
                f_a = f_alpha
                break

            if mu < 2*alpha - alpha_prev:
                alpha_prev = alpha
                alpha = mu
            else:
                # TODO:
                # find alpha as in book, right now I use middle point
                _alpha = alpha
                alpha = 0.5*(2*alpha - alpha_prev +
                         min(mu, alpha+self.tau1*(alpha-alpha_prev)))
                alpha_prev = _alpha

        while True:
            # TODO:
            # Should do polynomial interpolation here as well
            alpha = 0.5*((a + self.tau2*(b - a)) + (b - self.tau3*(b - a)))
            f_alpha = f(alpha)

            # check if alpha satisfies condition 1. If not we need a smaller
            # value choose [a, alpha]
            if (f_alpha > f_0 + self.rho*alpha*f_grad_0 or
                    f_alpha >= f_a):
                #a = a
                b = alpha
            else:
                f_grad_alpha = f_grad(alpha)
                # alpha satisfies condition 1 check condition 2 and if true
                # return that alpha, otherwise we're too close,
                #choose [alpha, b]
                if norm(f_grad_alpha) <= -self.sigma*f_grad_0:
                    return alpha

                # changing order with respect to Fletcher to avoid saving a
                if (b - a)*f_grad_alpha >= 0:
                    b = a
                a = alpha
                # else:
                    #b = b

    def cubic_minimize(fa, fpa, fb, fbp, a, b):
        """
        Fits a cubic polynomial to the points and derivatives and returns it's
        minimum in the interval
        """

        # Transform to [0, 1] (derivatives change)
        fpa = fpa*(b-a)
        fpb = fpb*(b-a)

        # The interpolating polynomial is given by:
        # fa + fpa*z + eta*z^2 + xsi*z^3
        eta = 3*(fb - fa) - 2*fpa - fpb
        xsi = fpa + fpb - 2*(fb - fa)

        # find inflection points
        poly = array([fa, fpa, eta, xsi])
        roots = P.polyroots(P.polyder(poly))
        roots = roots[np.logical_and(roots>0, roots<1)]
        values = r_[fa, P.polyval(roots, poly), fb]
        
        return r_[0, roots, 1][argmin(values)]

class QuasiNewtonBroyden(NewtonInexactLine):    
    
    def __init__(self, *args, **kwargs):
        super(QuasiNewtonBroyden, self).__init__(*args, **kwargs)
        
    def update_hessian(self, x, delta, H, G):
        #print str(norm(inv(self.opt_problem.hessian(x))-H, 'fro'))
        #print x
        f_grad = self.opt_problem.gradient
        gamma = f_grad(x+delta) - f_grad(x)
        u = delta - dot(H, gamma)
        a = 1/dot(u, gamma)
        H = H + dot(outer(delta - dot(H, gamma), delta), H) / \
                dot(delta, dot(H, gamma))
        return H, None

    def find_direction(self, f_grad_x, H, G):
        return dot(H, f_grad_x)

class QuasiNewtonBroydenBad(QuasiNewtonBroyden):    
    
    def __init__(self, *args, **kwargs):
        super(QuasiNewtonBroydenBad, self).__init__(*args, **kwargs)
        
    def update_hessian(self, x, delta, H, G):
        f_grad = self.opt_problem.gradient
        gamma = f_grad(x+delta) - f_grad(x)
        H = H + outer((delta - dot(H, gamma)) / dot(gamma, gamma),
                      gamma)
        return H, None

            
def main():
    def rosenbrock(x):
        return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    def rosenbrock_grad(x):
        return array([-200*(x[1]-x[0]) -2*(1-x[0]),
                        200*(x[1]-x[0]) ])
    def F(x):
        return x[0]**2 + x[0]*x[1] + x[1]**2
    def F_grad(x):
        return array([2*x[0]+x[1],x[0]+2*x[1]])

    guess = [-1, 1]

    opt = OptimizationProblem(rosenbrock, 2)
    cn  = ClassicNewton(opt)
    print "\nClassicNewton.Optimize(...): \n"
    print cn.optimize(guess)
    cn  = NewtonExactLine(opt)
    print "\nNewtonExactLine.Optimize(...): \n"
    print cn.optimize(guess)
    cn = NewtonInexactLine(opt);
    print "\nNewtonInexact.Optimize(...): \n"
    print cn.optimize(guess)
    cn = QuasiNewtonBroyden(opt);
    print "\nQuasiNewtonBroyden.Optimize(...): \n"
    print cn.optimize(guess, True)
    cn = QuasiNewtonBroydenBad(opt);
    print "\nQuasiNewtonBroydenBad.Optimize(...): \n"
    print cn.optimize(guess)


if __name__ == '__main__':
    main()
