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
    Class for handling splines using the blossom method to caluclate
    points. Also provides interpolation and plotting. 
    """
    
    def __init__(self, xs, u=None,interpol = False):
        """
        If the knot sequence isn't specified then an equidistant knot
        sequence u \in [0,1] is set up.
        """
        if u is None:
            u = linspace(0, 1, len(xs)-2)
        self.u = r_[u[0], u[0] ,u , u[-1], u[-1]] #padding
        if(interpol):
            self.interpPoints=xs #We like to see the old points. 
            xs = self.interpolate(xs)
        else:
            self.interpPoints=None
        self.xs = xs
        
    
    def interpolate(self, xs):
        """
        When initilizing the spline, if interpol was set then the provided
        points are not interpreted as control points but as points through
        which the curve should pass. This method solves the resulting
        equation system. 
        Can be extended to provide a more efficient computation for
        equidistand knots. 
        """   
        p = (self.u[0:-2]+self.u[1:-1]+self.u[2:])/3.0
        a = zeros([len(xs),len(xs)])
        for i in range(len(xs)):
            for j in range(len(xs)):
                b = basisFunction(self.u[2:-1],j)
                a[i,j] = b(p[i])[0]
        return lg.solve(a,xs)

    def __coeff(self, minI, maxI, u, d, j):
        """
        Used to calculate an entry in the blossom algorithm.
        min/maxI - current index for the knot sequence.
        u - point where the curve is  calculated.
        d - vector with relevant entries in the blossom
        j - index to d-vector
        """
        try:
            alpha  = float(self.u[maxI] - u)/float(self.u[maxI]-self.u[minI])
        except ZeroDivisionError:
            return d[j+1]
        return alpha*d[j] + (1 - alpha)*d[j+1]
        
        
    
    def __call__(self, u):
        """
        Evaluates the spline at u. 
        Can handle vector form of u. 
        """
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
        """
        Evaluates the spline at a single point u. Help method to __call__
        I - u:s blossom index.
        """
        if u == self.u[0]:
            return self.xs[0]

        # initalize blossoms
        d  = self.xs[I-2:I+2].copy()

        for i in range(3):
            for j in range(0, 3-i):
                d[j] = self.__coeff(I + i - 2 + j ,
                               I + j + 1 , u, d, j)
        return d[0]
        
        
    def plot(self):
        """
        Basic plot method. Plots point and control points by default.
        If interpolation was choosen then the old points, interpolation
        points, are plotted as well.
        """
        over = linspace(self.u[0], self.u[-1], 200)
        points = self(over)
        plot(self.xs[:,0],self.xs[:,1], '*')
        if(self.interpPoints is not None):
            plot(self.interpPoints[:,0],self.interpPoints[:,1], '^')
        plot(points[:,0],points[:,1], '+')
        plot(points[:,0],points[:,1], linewidth=1)
        show()
        
def basisFunction(knots,j):
    """
    Constructs a spline which can be used to determine the 
    j:th basic function for the knot sequence knot.
    Returns a new Spline instance with the relevant unit vector set
    as control point. 
    """
    d = zeros([len(knots)+2,2])
    d[j] = array([1,1])
    return Spline(d,knots)
    
def test():
    """ Plots a few basic functions.
    """
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
    #a = array(mat('0.,0.; 1.,1.3; 2.,2.9; 3.,5.; 4.,2.; 3.,0.; 2.,-1.2;0.,0.'))
    a = array(mat('1.,0.;1.,2.;2.5,2.;0.5,4.5;1.5,7.5;0.5,2.75;-.5,7.50;0.5,4.5;-1.5,2.0;0.,2.;0.,.0'))
    
    s = Spline(a,interpol = True)
    s.plot()

if __name__ == '__main__':
    main()

 
