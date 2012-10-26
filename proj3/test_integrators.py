
from assimulo import testattr
from assimulo.explicit_ode import *
from assimulo.problem import Explicit_Problem, cExplicit_Problem
from assimulo.solvers import LSODAR
from assimulo.exception import *
import nose
from newmark import Newmark
from hht import HHT
from scipy import array
from numpy import zeros_like
from math import sqrt, exp, sin, cos

class mass_spring_damper(object):
    def __init__(self, mass, k, c):
        self.mass = float(mass)
        self.k = float(k)
        self.c = float(c)
        self.omega0 = sqrt(self.k/self.mass)
        self.zeta = c/(2*sqrt(self.mass*self.k))

    def __call__(self, y, yprime, t):
        return -self.omega0**2*y-2*self.zeta*self.omega0*yprime
    
    def explicit(self, t, yprime0, y0):
        if self.zeta == 1.0:
            return (y0+(yprime0+self.omega0*y0))*t*exp(-self.omega0*t)
        elif self.zeta > 1.0:
            raise NotImplemented("Overdamped System")
        else:
            omegad = self.omega0*sqrt(1-self.zeta**2)
            A = y0
            B = 1/omegad*(self.zeta*self.omega0*y0+yprime0)
            return (exp(-self.zeta*self.omega0*t)*
                    (A*cos(omegad*t)+B*sin(omegad*t)))



def test_newmark_basic_2nd_order_no_damping():
    f  = lambda t: 9.82/2.0 * t**2  + 5
    """ Take off in a rocket at g-acc from 5m above grounds """
    f2 = lambda t, x, y: 9.82
    y0 = 5.0
    v0 = 0.0
    prob = Explicit_Problem(f2, y0)
    nmark = Newmark(prob, v0)
    end = 10.0
    t, y = nmark.simulate(end)
    print y[-1], f(end), t[-1]
    nose.tools.assert_almost_equal(y[-1], f(end))

def test_newmark_basic_2nd_order_damping():
    f  = lambda t: 5.0/3.0* t**3 + 2*t + 3
    """ Take off in a rocket at g-acc from 5m above ground, v0 = 10m/s """
    def ode_2(x,y,t):
        return 10.0 * t
    y0 = 3.0
    v0 = 2.0
    prob = Explicit_Problem(ode_2, y0)
    vals = array([0.0,0.33,0.66,1.0])
    for beta in vals:
        for gamma in vals:                
            nmark = Newmark(prob, v0, beta=beta, gamma=gamma)
            end = 10.0
            t, y = nmark.simulate(end)
            nose.tools.assert_almost_equal(y[-1]/y[-1], f(end)/y[-1],places = 1)

def test_hht_basic_2nd_order_damping():
    f  = lambda t: 5.0/3.0* t**3 + 2*t + 3
    """ Take off in a rocket at g-acc from 5m above ground, v0 = 10m/s """
    #f2 = lambda x, y, t: 10.0 * t
    def ode_2(x,y,t):
        return 10.0 * t
    y0 = 3.0
    v0 = 2.0
    prob = Explicit_Problem(ode_2, y0)
    vals = array([-1.0/3.0,-0.2,-0.1,-0.05,0.0])
    for alpha in vals:              
        hht = HHT(prob, v0, alpha = alpha)
        end = 10.0
        t, y = hht.simulate(end)
        nose.tools.assert_almost_equal(y[-1]/y[-1], f(end)/y[-1],places = 1)

def test_hht_basic_2nd_order_spring():
    y0 = 3.0
    v0 = 2.0
    ode = mass_spring_damper(3, 2, 3)
    prob = Explicit_Problem(ode, y0)
    vals = array([-1.0/3.0,-0.2,-0.1,-0.05,0.0])
    for alpha in vals:              
        hht = HHT(prob, v0, alpha = alpha)
        end = 10.0
        t, y = hht.simulate(end)
        nose.tools.assert_almost_equal(y[-1]/y[-1], ode.explicit(end, v0, y0)/y[-1],places = 1)
        
class flattened_2nd_order(object):
    
    def __init__(self, mass, k, c):
        self.ode = mass_spring_damper(mass, k, c)
        
    def __call__(self, t, y):
        y_bis   = self.ode(y[0], y[1], t)
        y_prime = - y[0] * self.ode.omega0 ** 2.0 / (2.0*self.ode.zeta * self.ode.omega0) - y_bis
        
        return array([y_prime,y_bis])

    def explicit(self, t, yprime0, y0):
        return self.ode.explicit(t, yprime0, y0)
        
def test_against_normal_solvers():
    end = 10.0
    ode = flattened_2nd_order(3, 2, 3)
    y0 = 3.0
    v0 = 2.0
    prob = Explicit_Problem(ode, [y0,v0])
    exp_sim = LSODAR(prob)
    t, y = exp_sim.simulate(end)
    
    ode = mass_spring_damper(3, 2, 3)
    prob = Explicit_Problem(ode, y0)
    hht = HHT(prob, v0, alpha = -0.2)
    t, y_2 = hht.simulate(end)
    print y[-1][0], y_2[-1]
    nose.tools.assert_almost_equal(y[-1][0]/y[-1][0], y_2[-1]/y[-1][0],places = 1)

def test_truck_hht():
    from problems import Truck
    end = 10.0
    t = Truck()
    ode = t.fcn
    y0 = t.initial_conditions()
    v0 = zeros_like(y0)
    prob = Explicit_Problem(ode, y0)
    hht = HHT(prob, v0, alpha = -0.2)
    t, y = hht.simulate(end)
    for i in y0.size:
        plot(t, y[:,i])

    show()


def test_pend_agains_normal():
    import problems as p
    from matplotlib.pylab import plot, show
    pend = p.Pendulum2nd()
    ode  = pend.fcn
    end = 15
    y0   = pend.initial_condition()
    prob = Explicit_Problem(ode, y0[0])
    hht = HHT(prob, y0[1], alpha=-0.2)
    t, y_1 = hht.simulate(end)
    
    ode  = pend.fcn_1
    prob = Explicit_Problem(ode, y0)
    sim = LSODAR(prob)
    t, y_2 = hht.simulate(end)
    
    nose.tools.assert_almost_equal(y_2[-1][0]/y_2[-1][0], y_1[-1]/y_2[-1][0],places = 1)
    
    

