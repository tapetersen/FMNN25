#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy V
#  

from  __future__  import division
from  scipy       import *
from  matplotlib.pyplot import *
import spline as sp

class Tester(object):
    

    def __deCasteljau(self,cp,t):
        bi = array(cp)
        for j in range(len(cp)):
            for i in range(len(cp)-(j+1)):
                bi[i] = (1. - t)*bi[i] + t*bi[i+1]
        return bi[0]

    def test_SumsUpToOne(self):
        # the sum of the basis functions should add up to one
        s = zeros([10,2])
        u = linspace(0,1,10)
        for i in range(10):
            for j in range(12):
                b = sp.basisFunction(u,j)
                s[i] = s[i] + b(u[i])
        b = abs(s[:,:] - 1.0) < finfo(double).eps #machine epsilon
        print b,s,sum(b)
        assert(sum(b)==20)
        
    def test_sameFromBlossomAsFromDeCastelja(self):
        cp = array([[0.,0.1], [0.3,0.6], [0.7,0.8],[1.0,0.2]])
        s = sp.Spline(cp)
        print s(0.5), self.__deCasteljau(cp,0.5)
        k = self.__deCasteljau(cp,0.5)
        j = s(0.5)
        k[:] = abs(k[:]-j[:]) < finfo(double).eps #machine epsilon
        assert(sum(k) == 2)

    def test_vandermonde(slef):
        # check that the vandermonde has only 4 different values for 
        # en equidistant grid
        pass

    def test_samefromblossomasfromrecursion(self):
        pass
