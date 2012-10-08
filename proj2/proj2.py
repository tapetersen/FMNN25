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
from collections import defaultdict
import chebyquad as cqp

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
    
    def __init__(self, function, 
                 gradient = False, hessian = False):
        """
        The constructor receives information abut what kind of transform is needed
        for the moment - only one transform can be specified.

        Exceptions:
            - If no transforms are specified the class asks the user for one
            - If two transforms are specfied the class asks for only one to be specified,
            the transform can only be uniquely specified.
        """
        if not (gradient or hessian):
            raise Exception("you must specify a transform")
        elif gradient and hessian:
            raise Exception("You can only specify one transform");
        self.grad = gradient
        self.hess = hessian
        self.f    = function


    def gradient(self, x):
        x = asarray(x)
        grad = zeros(x.size)
        h    = 1e-5 
        for i in range(x.size):
            step    = zeros(x.size)
            step[i] = h
            grad[i] = (self.f(x+step) - self.f(x-step))/(2.*h)

        return grad

    def gradient2(self, x, fx=None):
        """Approximates the gradient using (central) finite differences 
        of degree 1, or forwards if fx is supplied
        """
        # force column vector
        x = asarray(x).reshape(-1, 1)

        ## hh is matrix with h in diagonal
        h    = 1e-5 
        hh   = eye(x.size)*h

        # x+hh will be matrix with x+xi*h in the column vectors
        if fx is None:
            return (self.f(x+hh) - self.f(x-hh))/(2*h)
        else:
            return (self.f(x+hh) - fx)/h


    def hessian(self, x):
        """ Approximates the hessian using (central) finite differences of degree 2
        A symmtrizing step: hessian = .5*(hessian + hessian^T) is
        also performed
        """
        x = asarray(x)
        hess = zeros((x.size, x.size))
        h    = 1e-5
        # Approximates hessian using gradient, see 
        # http://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
        # TODO: We don't need to compute this many values since its
        # symmetric. If we do t more efficiently we don't need
        # the symmetrizing step (I think). - B
        for i in range(x.size):
            for j in range(x.size):
                step1     = zeros_like(x)
                step2     = zeros_like(x)
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
        """
        Evaluation function that performs the transfrm specfied,
        the constructuor ensures that the transform is uniquely determind at instaciation.
        """
        if self.grad:
            return self.gradient(x)
        if self.hess:
            return self.hessian(x)
        #raise Exception("Transform incompletely specified")
        #This eception is never reached since always one trasform is guaranteed to be specified through the constructor 

class OptimizationProblem(object):
    """ Provides an interface to various methods on a given function
    which can be used to optimize it. 
    """
    
    def __init__(self, objective_function,
                            function_gradient = None):
        """The user provides a function, the functions dimension (\in R^n) and
        optionally its gradient (given as a callable function)

        An atribute 'is_function_gradient' is added ss a boolean so that we can keep track
        if we're working with a matrix or a function
        
        """
        
        self.objective_function = objective_function
        self.is_function_gradient = False;
        """
            A gradient is specfied by the user, use it.
            Otherwise - obtain the gradient numerically

            Always construct the Hessian numerically
        """
        if function_gradient is not None:
            self.gradient = function_gradient
            self.is_function_gradient = True;
        else:
            self.gradient = FunctionTransforms(objective_function, 
                                            gradient=True)
        self.hessian = FunctionTransforms(objective_function,  
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
        

    def optimize(self, guess, debug=False):
        x = asarray(guess)

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

            if norm(f_grad_x) < 1e-4 :
                break
        
            direction = self.find_direction(f_grad_x, H, G)
            #assert dot(-direction, f_grad_x) < 0:
            alpha = self.find_step_size(
                f=lambda alpha: f(x - alpha*direction),
                f_grad=lambda alpha: dot(f_grad(x - alpha*direction), -direction))
            
            delta = -alpha*direction
            H, G = self.update_hessian(x, delta, H, G) 
            print "H: ", H
            print "G: ", G
            x = x + delta
            f_grad_x = f_grad(x)

        if debug:
            self.xs = array(self.xs)
            print self.xs
            self.plot()

        return x

    def find_step_size(self, f, f_grad):
        return 1

    def find_direction(self, f_grad_x, H, G):
        try:
            factored = lg.cho_factor(G)
            return lg.cho_solve(factored, f_grad_x)
        except LinAlgError:
            raise LinAlgError(
                "Hessian not positive definite, converging to saddle point")

    def update_hessian(self, x, delta, H, G):
        return (None, self.opt_problem.hessian(x + delta))

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

        alpha = min(1., mu*.25)
        alpha_prev = 0.

        f_alpha = f_0
        f_grad_alpha = f_grad_0
        # Begin the bracketing phase
        for it in range(100):
            f_prev_alpha = f_alpha
            f_grad_prev_alpha = f_grad_alpha
            f_alpha = f(alpha)

            if f_alpha < self.minimum_bound:
                return alpha

            # check condition 1 or if new f value is higher
            if (f_alpha > f_0 + alpha*self.rho*f_grad_0 or
                    f_alpha >= f_prev_alpha):
                a = alpha_prev
                b = alpha
                f_a = f_prev_alpha
                f_b = f_alpha
                f_grad_a = f_grad_prev_alpha
                break

            # check condition 2
            f_grad_alpha = f_grad(alpha)
            if norm(f_grad_alpha) <= -self.sigma*f_grad_0:
                return alpha

            if f_grad_alpha >= 0:
                a = alpha
                b = alpha_prev
                f_b = f_prev_alpha
                f_a = f_alpha
                f_grad_a = f_grad_alpha
                break

            if mu < 2*alpha - alpha_prev:
                alpha_prev = alpha
                alpha = mu
            else:
                _alpha = alpha
                left = 2*alpha - alpha_prev
                right = min(mu, alpha+self.tau1*(alpha-alpha_prev))
                alpha = cubic_minimize(
                    f(left), f_grad(left),
                    f(right), f_grad(right),
                    left, right)
                alpha_prev = _alpha
        else:
            # bracketing failed horribly
            print "Warning, couldn't bracket alpha defaulting to 1.0"
            return 1.0

        # check conditions in book (and that the cached values are correct)
        assert f_a == f(a)
        assert f_a <= f_0 + a*self.rho*f_grad_0
        assert f_grad_a == f_grad(a)
        assert (b-a)*f_grad_a<0
        assert f_b == f(b)
        assert f_b > f_0 + b*self.rho*f_grad_0 or f_b >= f_a

        for it in range(50):
            left = a + self.tau2*(b - a)
            right = b - self.tau3*(b - a)
            # don't interpolate if too small
            if abs(left-right) > 1e-5:
                alpha = cubic_minimize(
                    f(left), f_grad(left),
                    f(right), f_grad(right),
                    left, right)
            else:
                alpha = (left + right)*0.5

            # check if alpha satisfies condition 1. If not we need a smaller
            # value choose [a, alpha]
            f_alpha = f(alpha)
            if (f_alpha > f_0 + self.rho*alpha*f_grad_0 or
                    f_alpha >= f_a):
                #a = a
                b = alpha
            else:
                # alpha satisfies condition 1 check condition 2 and if true
                # return that alpha, otherwise we're too close,
                #choose [alpha, b]
                f_grad_alpha = f_grad(alpha)
                if norm(f_grad_alpha) <= -self.sigma*f_grad_0:
                    return alpha

                # changing order with respect to Fletcher to avoid saving a
                if (b - a)*f_grad_alpha >= 0:
                    b = a
                a = alpha
                # else:
                    #b = b
        else:
            # Sectioning failed horribly
            print "Warning, couldn't find acceptable alpha defaulting to 1.0"
            return 1.0


class QuasiNewtonDFP(NewtonInexactLine):    
    """ Implemented as in lecture 3 """
    
    def __init__(self, *args, **kwargs):
        super(QuasiNewtonDFP, self).__init__(*args, **kwargs)
        
    def update_hessian(self, x, delta, H, G):
        f_grad = self.opt_problem.gradient
        gamma = f_grad(x+delta) - f_grad(x)
        d_dot_g = dot(delta, gamma)
        H_dot_g = dot(H, gamma)
        H = H + (outer(delta, delta)/d_dot_g - 
                 dot(outer(H_dot_g, gamma), H)/dot(gamma, H_dot_g))

        return H, None

    def find_direction(self, f_grad_x, H, G):
        return dot(H, f_grad_x)

class QuasiNewtonBFSG(NewtonInexactLine):    
    """ Implemented as on  http://en.wikipedia.org/wiki/BFGS_method """
    
    def __init__(self, *args, **kwargs):
        super(QuasiNewtonBFSG, self).__init__(*args, **kwargs)
        
    def update_hessian(self, x, delta, H, G):
        f_grad = self.opt_problem.gradient
        gamma = f_grad(x+delta) - f_grad(x)
        d_dot_g = dot(delta, gamma)
        H_dot_g = dot(H, gamma)
        H = H + ( (d_dot_g+dot(gamma, H_dot_g))*
                    outer(delta, delta)/d_dot_g**2 
                 -
                (outer(H_dot_g, delta) + dot(outer(delta, gamma), H))/
                d_dot_g)

        return H, None

    def find_direction(self, f_grad_x, H, G):
        return dot(H, f_grad_x)

class QuasiNewtonBroyden(NewtonInexactLine):    
    
    def __init__(self, *args, **kwargs):
        super(QuasiNewtonBroyden, self).__init__(*args, **kwargs)
        
    def update_hessian(self, x, delta, H, G):
        #print str(norm(inv(self.opt_problem.hessian(x))-H, 'fro'))
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
        u = delta-dot(H, gamma)
        a = 1/dot(u, gamma);
        H = H + a*outer(u, u)
        return H, None

            
def cubic_minimize(fa, fpa, fb, fpb, a, b):
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
    poly = poly1d([xsi, eta, fpa, fa])

    # find inflection points
    if poly.order == 3:
        der = poly.deriv()
        if der[2]:
            der = der/der[2]
            disc = (der[1]*der[1]/4 - der[0])
            if disc > 0:
                points = asarray(
                    [0.0 ,-der[1]/2+sqrt(disc), -der[1]/2-sqrt(disc), 1.0])
                points = points[np.logical_and(points >= 0, points <= 1)]
            else:
                # can't have minimums
                points = asarray([0.0, 1.0])
        else:
            # can't have minimums
            points = asarray([0.0, 1.0])
    elif poly.order == 2:
        extreme = -poly[1]/(2*poly[2])
        if extreme < 1 and extreme > 0:
            points = asarray([0., extreme, 1.])
        else:
            points = asarray([0.0, 1.0])

    else:
        points = asarray([0.0, 1.0])

    values = poly(points)
    return points[argmin(values)]*(b-a)+a

# Test functions
def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
def rosenbrock_grad(x):
    return array([-200*(x[1]-x[0]) -2*(1-x[0]),
                    200*(x[1]-x[0]) ])
def F(x):
    return x[0]**2 + x[0]*x[1] + x[1]**2
def F_grad(x):
    return array([2*x[0]+x[1],x[0]+2*x[1]])

def main():

    def f(x):
        return (x[0]+1)**2 + (x[1]-1)**2
    guess = array([-1.0,1.0])

    op = OptimizationProblem(rosenbrock)
    #cn  = ClassicNewton(op)
    #print "\nClassicNewton.Optimize(...): \n"
    #print cn.optimize(guess)
    #cn  = QuasiNewtonBFSG(op)
    #print "\nBFSG.Optimize(...): \n"
    #print cn.optimize(guess)
    #cn = NewtonInexactLine(op);
    #print "\nNewtonInexact.Optimize(...): \n"
    #print cn.optimize(guess)
    #cn = QuasiNewtonBroyden(op);
    #print "\nQuasiNewtonBroyden.Optimize(...): \n"
    #print cn.optimize(guess, True)
    #cn = QuasiNewtonBFSG(op)
    #print "\nQuasiNewtonBFSG.Optimize(...): \n"
    #print cn.optimize(guess, True)
    #cn = QuasiNewtonDFP(op)
    #print "\nQuasiNewtonDFP.Optimize(...): \n"
    #print cn.optimize(guess, True)
    #return 
    cn = QuasiNewtonBroydenBad(op);
    print "\nQuasiNewtonBroydenBad.Optimize(...): \n"
    print cn.optimize(guess, True)


if __name__ == '__main__':
    main()
