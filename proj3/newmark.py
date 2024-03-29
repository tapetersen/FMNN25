
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
from scipy.optimize import fsolve
ID_COMPLETE = 3


class Newmark(Explicit_ODE):
    """ Solves a second order ODE using HHT method.
    Takes an instance of a problem (Explicit problem) 
    and initial conditions for yprime
    """
    
    def __init__(self, problem, v0, beta=None, gamma=None):
        """ Initialize solver, check if explicit, store functions etc """
        super(Newmark, self).__init__(problem)
        self.solver = fsolve
        self.options["h"] = 0.01
        self.f  = problem.rhs
        self.v = v0
        if(beta is not None and gamma is not None):  # Check if the solver is explicit
            self.explicit = False
            self.beta  = beta
            self.gamma = gamma
        else:
            self.explicit = True #No damping
    
    def _step(self, t, y, h):
        """ Used to take a step in the integrate method while
        simulating"""
        if(self.explicit):
            y = y + h*self.v + (h**2 / 2.0) * self.f(t,y,self.v)
            a = self.f(t,y,self.v)
            self.v = self.v + h/2.0 * (self.a + a)
            self.a = a
            return t+h,y
        else: # We must use solvers / implicit form. 
            f_pn1 = lambda a_n1:  (y + h*self.v + (h**2 / 2.0) * \
                           ((1.0 - 2.*self.beta)*self.a + 2.*self.beta*a_n1))
            f_vn1 = lambda a_n1:  (self.v + h *((1.0-self.gamma)*self.a + self.gamma*a_n1))
            f_an1 = lambda a_n1: a_n1 - (self.f(t+h,f_pn1(a_n1),f_vn1(a_n1)))
            
            a = fsolve(f_an1, self.a) # solve the residual equation w.r.t. a_n+1
            
            y      = f_pn1(a) # Calculate and store new variables. 
            self.v = f_vn1(a)
            self.a = a
            return t+h, y
            
    def integrate(self, t, y, tf,  opts):
        """ Main method used when simulating
        Takes step with constant stepsize until the final time
        is reached."""
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


