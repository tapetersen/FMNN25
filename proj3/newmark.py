
from assimulo import testattr
from assimulo.explicit_ode import *
from assimulo.problem import Explicit_Problem, cExplicit_Problem
from assimulo.exception import *
import numpy as N
import pylab as P
from assimulo.solvers import CVode
import nose
from scipy import double


class Newmark(Explicit_ODE):
    
    def __init__(self, problem, beta=None, gamma=None):
		if(beta==None or gamma==None): #No damping
			self.explicit = True
		else:
			self.beta  = beta
			self.gamma = gamma
		
    def step(self,double t,N.ndarray y,double tf,dict opts):
        cdef double h
        h = self.options["h"]
        
        if t+h < tf:
            t, y = self._step(t,y,h)
            return ID_OK, t, y
        else:
            h = min(h, abs(tf-t))
            t, y = self._step(t,y,h)
            return ID_COMPLETE, t, y
    
    def _step(self, t, y, h):
		if(self.explicit):
			y = y + h*self.v + (h**2 / 2.0) + self.a
			a = self.problem.rhs(t,y)
			self.v = self.v + h/2.0 * (a + self.a)
			self.a = a
    
    def integrate(self, double t,N.ndarray y,double tf, dict opts):
        cdef double h
        cdef list tr,yr
        
        h = self.options["h"]
        h = min(h, abs(tf-t))
        
        tr = []
        yr = []
        
        while t+h < tf:
            t, y = self._step(t,y,h)
            tr.append(t)
            yr.append(y)
            h=min(h, abs(tf-t))
        else:
            t, y = self._step(t, y, h)
            tr.append(t)
            yr.append(y)
        
        return ID_COMPLETE, tr, yr



