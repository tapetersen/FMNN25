
from assimulo import testattr
from assimulo.explicit_ode import *
from assimulo.problem import Explicit_Problem, cExplicit_Problem
from assimulo.exception import *
import numpy as N
import pylab as P
from assimulo.solvers import CVode
import nose
from scipy import double


class Newmark(Explicit_Problem):
    
    def __init__(self, rhs=None, y0=None, t0=0.0, p0=None, 
                 sw0=None):
        super(cExplicit_Problem, self).__init__(y0, t0, p0, sw0)       
        if rhs != None:
            self.rhs = rhs
    
    def handle_result(self, solver, t, y):
        #import scipy.optimize as so
        #xmin= so.fmin_bfgs(chebyquad,x,gradchebyquad) 
        """
        Method for specifying how the result is to be handled. As default the
        data is stored in two vectors: solver.(t/y).
        """
        i = 0
        solver.t_sol.extend([t])
        solver.y_sol.extend([y])
       
        ##Store sensitivity result (variable _sensitivity_result are set from the solver by the solver)
        if self._sensitivity_result == 1:
            for i in range(solver.problem_info["dimSens"]):
                solver.p_sol[i] += [solver.interpolate_sensitivity(t, i=i)]


def run_example(with_plots=True):
    hello = "hellooo\n"
    print hello
    #Define the rhs
    def f(t,y):
        ydot = -y[0]
        return N.array([ydot])
    
    #Define an Assimulo problem
    exp_mod = Newmark(f, y0=4.0)
    exp_mod.name = 'Simple CVode Example'
    print hello
    #Define an explicit solver
    exp_sim = CVode(exp_mod) #Create a CVode solver
    print hello
    #Sets the parameters
    exp_sim.iter  = 'Newton' #Default 'FixedPoint'
    exp_sim.discr = 'BDF' #Default 'Adams'
    exp_sim.atol = [1e-4] #Default 1e-6
    exp_sim.rtol = 1e-4 #Default 1e-6
    print hello
    #Simulate
    t1, y1 = exp_sim.simulate(5,100) #Simulate 5 seconds
    t2, y2 = exp_sim.simulate(7) #Simulate 2 seconds more
    print hello
    #Basic test
    #nose.tools.assert_almost_equal(y2[-1], 0.00347746, 5)
    
    #Plot
    if with_plots:
        P.plot(t1, y1, color="b")
        P.plot(t2, y2, color="b")
        P.show()
