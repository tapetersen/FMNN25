# -*- coding: utf-8 -*-
"""
Chebyquad Testproblem

Course Material for the course FMNN25

Created on Wed Nov 23 22:52:35 2011

@author: Claus Führer
"""
from  __future__  import division
from  scipy       import dot,linspace
from  scipy.special import eval_chebyt, eval_chebyu
import scipy.optimize as so
from numpy import array


def T(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the first kind
    x evaluation point (scalar)
    n degree 
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x
    return 2. * x * T(x, n - 1) - T(x, n - 2)

def U(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the second kind
    x evaluation point (scalar)
    n degree 
    Note d/dx T(x,n)= n*U(x,n-1)  
    """
    if n == 0:
        return 1.0
    if n == 1:
        return 2. * x
    return 2. * x * U(x, n - 1) - U(x, n - 2) 
    
def chebyquad_fcn_old(x):
    """
    Nonlinear function: R^n -> R^n
    """    
    n = len(x)
    def exact_integral(n):
        """
        Generator object to compute the exact integral of
        the transformed Chebychev function T(2x-1,i), i=0...n
        """
        for i in xrange(n):
            if i % 2 == 0: 
                yield -1./(i**2 - 1.)
            else:
                yield 0.

    exint = exact_integral(n)
    
    def approx_integral(i):
        """
        Approximates the integral by taking the mean value
        of n sample points
        """
        return sum(T(2. * xj - 1., i) for xj in x) / n
    return array([approx_integral(i) - exint.next() for i in xrange(n)]) 

def chebyquad_fcn(x):
    """
    Nonlinear function: R^n -> R^n
    """    
    n = len(x)
    def exact_integral(n):
        """
        Generator object to compute the exact integral of
        the transformed Chebychev function T(2x-1,i), i=0...n
        """
        for i in xrange(n):
            if i % 2 == 0: 
                yield -1./(i**2 - 1.)
            else:
                yield 0.

    exint = exact_integral(n)
    
    def approx_integral(i):
        """
        Approximates the integral by taking the mean value
        of n sample points
        """
        return sum(eval_chebyt(i, 2. * xj - 1.) for xj in x) / n
    return array([approx_integral(i) - exint.next() for i in xrange(n)]) 

def chebyquad_old(x):
    """            
    norm(chebyquad_fcn)**2                
    """
    chq = chebyquad_fcn_old(x)
    return dot(chq, chq)

def chebyquad(x):
    """            
    norm(chebyquad_fcn)**2                
    """
    chq = chebyquad_fcn(x)
    return dot(chq, chq)

def gradchebyquad_old(x):
    """
    Evaluation of the gradient function of chebyquad
    """
    chq = chebyquad_fcn_old(x)
    UM = 4. / len(x) * array([[(i+1) * U(2. * xj - 1., i) 
                             for xj in x] for i in xrange(len(x) - 1)])
    return dot(chq[1:].reshape((1, -1)), UM).reshape((-1, ))

def gradchebyquad(x):
    """
    Evaluation of the gradient function of chebyquad
    """
    chq = chebyquad_fcn(x)
    UM = 4. / len(x) * array([[(i+1) * eval_chebyu(i, 2. * xj - 1.) 
                             for xj in x] for i in xrange(len(x) - 1)])
    return dot(chq[1:].reshape((1, -1)), UM).reshape((-1, ))
    
if __name__ == '__main__':
    x=linspace(0,1,8)
    xmin= so.fmin_bfgs(chebyquad,x,gradchebyquad)  # should converge after 18 iterations  
    print xmin
    xmin= so.fmin_bfgs(chebyquad_old,x,gradchebyquad_old)  # should converge after 18 iterations  
    print xmin
