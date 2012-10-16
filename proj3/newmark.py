
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
import scipy.optimize as so
ID_COMPLETE = 3

class Newmark(Explicit_ODE):
    
    def __init__(self, problem, v0, beta=None, gamma=None):
        super(Newmark, self).__init__(problem)
        self.options["h"] = 0.01
        self.f  = problem.rhs
        self.v = v0
        if(beta is not None and gamma is not None): 
            self.explicit = False
            self.beta  = beta
            self.gamma = gamma
        else:
            self.explicit = True #No damping
    
    def _step(self, t, y, h):
        if(self.explicit):
            y = y + h*self.v + (h**2 / 2.0) * self.f(t,y,self.v)
            a = self.f(t,y,self.v)
            self.v = self.v + h/2.0 * (self.a + a)
            self.a = a
            return t+h,y
        else: # We must use solvers / implicit form
            f = lambda p: p - (y + h*self.v + (h**2 / 2.0) * \
                           ((1.0 - 2.*self.beta)*self.a + 2.*self.beta*self.f(t+h,p,self.v)))
            y = so.fmin_bfgs(f,y)
            f = lambda v: v - (self.v + h *((1.0-self.gamma)*self.a + self.gamma*(self.f(t+h,y,v))))
            self.v = so.fmin_bfgs(f,self.v)
            self.a = self.f(t+h,y,self.v)
            return t+h, y
            
    def integrate(self, t, y, tf,  opts):
        h = self.options["h"]
        h = min(h, abs(tf-t))
        tr = []
        yr = []
        self.a = self.f(t,y,self.v)
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


