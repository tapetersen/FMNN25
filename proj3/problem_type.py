
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


def run_example(with_plots=True):
    
    #Define the rhs
    def f(t,y):
        ydot = -y[0]
        return N.array([ydot])
    
    #Define an Assimulo problem
    exp_mod = Newmark(f, y0=4.0)
    exp_mod.name = 'Simple CVode Example'
    
    #Define an explicit solver
    exp_sim = CVode(exp_mod) #Create a CVode solver
    
    #Sets the parameters
    exp_sim.iter  = 'Newton' #Default 'FixedPoint'
    exp_sim.discr = 'BDF' #Default 'Adams'
    exp_sim.atol = [1e-4] #Default 1e-6
    exp_sim.rtol = 1e-4 #Default 1e-6
    
    #Simulate
    t1, y1 = exp_sim.simulate(5,100) #Simulate 5 seconds
    t2, y2 = exp_sim.simulate(7) #Simulate 2 seconds more
    
    #Basic test
    #nose.tools.assert_almost_equal(y2[-1], 0.00347746, 5)
    
    #Plot
    if with_plots:
        P.plot(t1, y1, color="b")
        P.plot(t2, y2, color="r")
        P.show()
