#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy V
#  

from  __future__  import division
from  scipy       import *
from  matplotlib.pyplot import *
import spline as sp

class Tester(object):
    
    #def __init__(self):
        #pass
        
    #def testSu(self):
        #assert('b' == 'b')

	def test_SumsUpToOne(self):
		s = zeros([10,2])
		u = linspace(0,1,10)
		for i in range(10):
			for j in range(12):
				b = sp.basisFunction(u,j)
				s[i] = s[i] + b(u[i])
		b = abs(s[:,:] - 1.0) < 10.0**-15
		print b,s,sum(b)
		assert(sum(sum(b))==20)
		# the sum of the basis functions should add up to one
		

	def test_vandermonde(slef):
		# check that the vandermonde has only 4 different values for 
		# en equidistant grid
		pass

	def test_samefromblossomasfromrecursion(self):
		pass
