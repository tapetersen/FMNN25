
from assimulo import testattr
from assimulo.explicit_ode import *
from assimulo.problem import Explicit_Problem, cExplicit_Problem
from assimulo.exception import *
import numpy as N
import pylab as P
from assimulo.explicit_ode import Explicit_ODE
import nose
from scipy import double
from assimulo.ode import *
ID_COMPLETE = 3

class Newmark(Explicit_ODE):
    
    def __init__(self, problem, v0, beta=None, gamma=None):
        super(Newmark, self).__init__(problem)
        self.options["h"] = 0.01
        self.f  = problem.rhs
        self.yd1 = N.array([0.0]*len(self.y0))
        self.v = v0
        if(beta==None or gamma==None): #No damping
            self.explicit = True
        else:
            self.beta  = beta
            self.gamma = gamma
    
    def _step(self, t, y, h):
        if(self.explicit):
            y = y + h*self.v + (h**2 / 2.0) * self.f(t,y,self.v)
            a = self.f(t,y,self.v)
            self.v = self.v + h/2.0 * (self.a + a)
            self.a = a
            return t+h,y
    
    def integrate(self, t, y, tf,  opts):
        h = self.options["h"]
        h = min(h, abs(tf-t))
        tr = []
        yr = []
        self.a = self.f(t,y,self.v)
        print self.a
        
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

    def print_statistics(self, k):
        pass 


