
from assimulo import testattr
from assimulo.explicit_ode import *
from assimulo.problem import Explicit_Problem, cExplicit_Problem
from assimulo.exception import *
import nose
from newmark import Newmark
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
    f  = lambda t: 9.82/2.0 * t**2 + 10*t + 5
    """ Take off in a rocket at g-acc from 5m above ground, v0 = 10m/s """
    f2 = lambda t, x, y: 9.82
    y0 = 5.0
    v0 = 10.0
    prob = Explicit_Problem(f2, y0)
    vals = array([0.0,0.25,0.5,0.75,1.0])
    for beta in vals:
        for gamma in vals:                
            nmark = Newmark(prob, v0, beta=beta, gamma=gamma)
            end = 10.0
            t, y = nmark.simulate(end)
            nose.tools.assert_almost_equal(y[-1], f(end))
