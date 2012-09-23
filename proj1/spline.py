#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy V
#  

from  __future__  import division
from  scipy       import *
from  matplotlib.pyplot import *
import sys
import copy 
class Spline(object):
    """
    Sets up an equidistant knot sequence
    """
    def __init__(self, xs, u=None,interpol = False):
        if u is None:
            u = linspace(0, 1, len(xs)-2)
        self.u = r_[u[0],u[0],u,u[-1],u[-1]]
        if(interpol):
            xs = interpolate(xs)
        self.xs = xs
        
    def interpolate(self,xs):
        

    def __coeff(self,minI,maxI,u,d, j):
        try:
            alpha  = float(self.u[maxI] - u)/float(self.u[maxI]-self.u[minI])
        except ZeroDivisionError:
            return d[j+1]
        #alpha = 1- u
        #print "alpha = Umax - u / Umax - Umin " + " = " + \
        #"(" + str(self.u[maxI]) + " - " + str(u) + ")/" + "(" +\
        #str(self.u[maxI]) + " - " + str(self.u[minI]) + ") = " + str(alpha)
        #print "d is before: " + str(d)
        #print "j = " + str(j)
        #print "alpha * d[j] = " + str(alpha) +"*" + str(d[j]) + " = " + str(alpha*d[j])
        #print "(1-alpha) * d[j+1] = " + str(1-alpha) +"*" + str(d[j+1]) + " = " + str((1-alpha)*d[j+1])
        return alpha*d[j] + (1 - alpha)*d[j+1]
        
        
    """
    Evaluates the spline at u. 
    """
    def __call__(self, u):
        I  = self.__hot(u)
        d  = array(self.xs[I-2:I+2])
        print "u: " + str(u)
        print "Index: " + str(I)
        for i in range(3):
            for j in range(0, 3-i):
                #print "\n(minI,maxI) = " + str([2*i + j - 2 + I,j + 1 + i + I])
                k = self.__coeff(2*i + j - 2 + I,
                               j + 1 + i + I, u, copy.copy(d), j)
                #print "k: " + str(k)  
                d[j] = array(k)
                #if(d[j,0] < 0):
                    #raise Exception
                #print "d is after: " + str(d)
        return d[0]
        
        
    def plot(self):
        over = linspace(.01,.99,200)
        points = empty([len(over),2])
        for i in range(len(over)):
             points[i] = self(over[i])
        plot(self.xs[0,:],self.xs[1,:],'*')
        plot(points[:,0],points[:,1],'-')
        show()
        
        
    """
    Couldn't find a fast built in function, is there one? 
    """
    def __hot(self,u):
        # Check if u is in the interval, otherwhise return a value
        # indicating an error.
        if(u < self.u[0] or u> self.u[-1]):
            return -1
        for (i, ui) in enumerate(self.u):
            if(ui > u):
                return i-1
    
def basisFunction(knots,j):
    d = empty([len(knots)-2])
    d[:] = 0
    d[j] = 1
    return Spline(d,knots)
    
def test():
    t = linspace(.001,.9,55)
    points = empty([len(t),2])
    for i in range(0,10):
        b = basisFunction(linspace(0,1,100),i*10)
        for i in range(len(t)):
            points[i] = b(t[i])
        plot(points[:,0],points[:,1])
    show()


def main():
    a = array([[0.,0.], [1.,2.], [1.5,1.5],[1.75,1.5],[2.,1.], [3.,0.]  ])
    s = Spline(a)
    s.plot()

if __name__ == '__main__':
    main()

 
