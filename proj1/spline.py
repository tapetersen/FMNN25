#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy Lundberg
#  

from  __future__  import division
from  scipy       import *
from  scipy import linalg as lg
from  matplotlib.pyplot import *
import sys

class Spline(object):
    """
    Sets up an equidistant knot sequence
    """
    def __init__(self, xs, u=None,interpol = False):
        self.equi = u is None
        if u is None:
            u = linspace(0, 1, len(xs)-2)
        self.u = r_[u[0], u[0] ,u , u[-1], u[-1]]
        if(interpol):
            xs = self.interpolate(xs)
        self.xs = xs
        
    """
    When initilizing the spline, if interpol was set then the provided
    points are not interpreted as control points but as points through
    which the curve should pass. This method solves the resulting
    equation system. 
    """    
    def interpolate(self, xs):
        # TODO: check assumption on the size of u, we need at least 3
        # points i'd guess
        b_x = xs[:,0]
        b_y = xs[:,1]
        # Check if we have an equidistant knot sequence, if yes then the
        # matrix can be computed more efficiently
        if(self.equi):
            # Add efficient computation here
            return xs
        else:
            p = (self.u + append(self.u[1:],[0.0])+ append(self.u[2:],[0.0,0.0]))/3.0
            a = empty([len(xs),len(xs)])
            for i in range(len(xs)):
                for j in range(len(xs)):
                    b = basisFunction(linspace(0, 1, len(xs)-2),j)
                    a[i,j] = b(p[i])
            print a
        ab_x = dot(a,b_x)
        ab_y = dot(a,b_y)
        d = empty([len(xs),2])            
        d[:,0] = lg.solve_banded([3, 4],ab_x,b_x)
        d[:,1] = lg.solve_banded([3, 4],ab_y,b_y)
        return d

    def __coeff(self, minI, maxI, u, d, j):
        try:
            alpha  = float(self.u[maxI] - u)/float(self.u[maxI]-self.u[minI])
        except ZeroDivisionError:
            return d[j+1]
        return alpha*d[j] + (1 - alpha)*d[j+1]
        
        
    """
    Evaluates the spline at u. 
    """
    def __call__(self, u):

        I = searchsorted(self.u, u)-1
        try:
            it = iter(u) # check if iterable
            result = empty((size(u), self.xs.shape[1]))
            for i, u_ in enumerate(u):
                result[i] = self.__eval(u_, I[i])
            return result
        except TypeError:
            return self.__eval(u, I)

    def __eval(self, u, I):
        if u == self.u[0]:
            return self.xs[0]

        # initalize blossoms
        d  = self.xs[I-2:I+2].copy()

        for i in range(3):
            for j in range(0, 3-i):
                d[j] = self.__coeff(I + i - 2 + j ,
                               I + j + 1 , u, d, j)
        if(float(d[0][0]) > 10**8):
			raise Exception()
        return d[0]
        
        
    def plot(self):
        over = linspace(self.u[0], self.u[-1], 200)
        points = self(over)
        plot(self.xs[:,0],self.xs[:,1], '*')
        plot(points[:,0],points[:,1], '+')
        plot(points[:,0],points[:,1], linewidth=1)
        show()
        
def basisFunction(knots,j):
    d = empty([len(knots)+2,2])
    d[:] = array([0,0])
    d[j] = array([1,1])
    return Spline(d,knots)
    
def test():
    t = linspace(0,1,200)
    points = empty([len(t),2])
    for i in range(5):
        b = basisFunction(linspace(0,1,10),i)
        for i in range(len(t)):
            points[i] = b(t[i])
        plot(t,points[:,0])
    show()


def main():
    #a = array([[0.,0.], [1.,2.], [1.5,1.5],[1.75,1.5],[2.,1.],[3.,0.]])
    a = array(mat('0.,0.; 1.,1.3; 2.,2.9; 3.,5.; 4.,2.; 3.,0.; 2.,-1.2;0.,0.'))
    s = Spline(a,interpol = True)
    s.plot()

if __name__ == '__main__':
    main()

 
