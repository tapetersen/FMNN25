
from assimulo import testattr
from assimulo.explicit_ode import *
from assimulo.problem import Explicit_Problem, cExplicit_Problem
from assimulo.exception import *
import nose
from newmark import Newmark
from hht import HHT
from scipy import array

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
