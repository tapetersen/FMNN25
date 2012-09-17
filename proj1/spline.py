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
	def __init__(self,start=0.0,stop=5.0, res=0.1):
		self.start = start
		self.stop  = stop
		self.res   = res
		self.knots = append([self.start,self.start],arange(\
							self.start,self.stop,self.res))
		self.knots = append(self.knots,[self.stop,self.stop])
		print self.knots

	"""
	Evaluates the spline at u. 
	"""
	def __call__(self, u):
		pass
		
	def plot(self):
		pass
		

"""
Makes an instance. If running from a terminal up to three numerical
arguments can be passes, e.g. > python spline.py 1.0,2.0,0.1, to override
the default values set by the __init__ for the spline class.
"""
def main():
	args = (str(sys.argv[1])).split(',')
	attr = []
	for i in range(len(args)-1):
		attr =  append(attr, float(args[i+1]))
	sp = Spline(*attr)
	return 0

if __name__ == '__main__':
	main()


















