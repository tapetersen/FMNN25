
from assimulo import testattr
from assimulo.explicit_ode import *
from assimulo.problem import Explicit_Problem, cExplicit_Problem
from assimulo.exception import *
import numpy as N
import pylab as P
from assimulo.explicit_ode import Explicit_ODE
import nose
from scipy import double
from scipy.optimize import newton,brent
from assimulo.ode import *
import solvers as solvers
ID_COMPLETE = 3


class HHT(Explicit_ODE):
    
    def __init__(self, problem, v0, alpha):
        super(HHT, self).__init__(problem)
        self.solver = solvers.QuasiNewtonBFSG(None)
        self.options["h"] = 0.01
        self.f  = problem.rhs
        self.v = v0
        if(alpha < -1.0/3.0 or alpha > 0):
            print alpha
            raise Exception("Non-valid alpha for HHT method")
        self.alpha = alpha
        self.beta  = ( (1.0 - alpha)/2.0 )**2
        self.gamma = (1.0 - 2.0 * alpha)/2.0
    
    def _step(self, t, y, h):        
        # We must use solvers / implicit form
        f_pn1 = lambda a_n1:  (y + h*self.v + (h**2 / 2.0) * \
                       ((1.0 - 2.*self.beta)*self.a + 2.*self.beta*a_n1))
        f_vn1 = lambda a_n1:  (self.v + h *((1.0-self.gamma)*self.a + self.gamma*a_n1))
        def f_an1(a_n1):
            f_n1 = self.f(f_pn1(a_n1),f_vn1(a_n1),t+h)
            f_n = self.f(y,self.v,t)
            return abs(a_n1 - ((1.0+self.alpha)*f_n1 - self.alpha*f_n))

        a = brent(f_an1)
        #f_an1 = lambda a_n1: (1.0+self.alpha)*self.f(f_pn1(a_n1),f_vn1(a_n1),t+h) - self.alpha*self.f(y,self.v,t)
        #self.solver.opt_problem = solvers.OptimizationProblem(f_an1)
        #a = self.solver.optimize(self.a)  # Minimize w.r.t. a_n+1
        
        
        y      = f_pn1(a) # Calculate and store new variables. 
        self.v = f_vn1(a)
        self.a = a
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


