#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias-Alex Petersen, Johnny Sjöberg, Andy V
#  

from  __future__  import division
from  scipy       import *
from  matplotlib.pyplot import *
import sys

class Spline(object):
    """
    Sets up an equidistant knot sequence
    """
    def __init__(self, xs, u=None):
        self.xs = append(append(array([xs[0],xs[0]]), xs, 0),array([xs[4],xs[4]]),0)
        if u is None:
            u = arange(0, 1, 1/len(xs))
        self.u = r_[u[0],u[0],u,u[-1],u[-1]]

    def __coeff(self,minI,maxI,u,d, j):
        try:
            alpha  = float(self.u[maxI] - u)/float(self.u[maxI]-self.u[minI])
        except ZeroDivisionError:
            return d[j+1]
        return alpha*d[j] + (1 - alpha)*d[j+1]
        
        
    """
    Evaluates the spline at u. 
    """
    def __call__(self, u):
        I  = self.__hot(u)
        d  = self.xs[I-2:I+2]
        for i in range(3):
            for j in range(0, 3-i):
                d[j] = self.__coeff(2*i + j - 2 + I,
                               j + 1 + i + I, u, d, j)
        return d[0]
        
    def plot(self):
        pass
        
        
    """
    Couldn't find a fast built in function, is there one? 
    """
    def __hot(self,u):
        # Check if u is in the interval, otherwhise return a value
        # indicating an error.
        if(u < self.u[0] or u> self.u[-1]):
            return -1
        for (i, ui) in enumerate(self.u):
            if(u < ui):
                return i-1
        

"""
Makes an instance. If running from a terminal up to three numerical
arguments can be passes, e.g. > python spline.py 1.0,2.0,0.1, to override
the default values set by the __init__ for the spline class.
"""
def main():
    a = c_[arange(5), arange(5,0,-1)]
    sp = Spline(a)
    print sp.xs, sp.u
    return 0

if __name__ == '__main__':
    main()

 
