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
        self.xs = c_[xs[0],xs[0],xs,xs[-1],xs[-1]];
        if u is None:
            self.u = arange(0, 1, 1/len(xs))
        self.u = r_[u[0],u[0],u,u[-1],u[-1]];

    """
    Evaluates the spline at u. 
    """
    def __call__(self, u):
        I = __hot(u)
        d = self.xs[I-2:I+3]
        for i in range(3):
            alpha = 

        pass
        
    def plot(self):
        pass
        

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

