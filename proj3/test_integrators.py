
from assimulo import testattr
from assimulo.explicit_ode import *
from assimulo.problem import Explicit_Problem, cExplicit_Problem
from assimulo.exception import *
import nose
from newmark import Newmark

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
    f  = lambda t: 9.82/2.0 * t**2  + 5
    """ Take off in a rocket at g-acc from 5m above grounds """
    f2 = lambda t, x, y: 9.82
    y0 = 5.0
    v0 = 0.0
    prob = Explicit_Problem(f2, y0)
    nmark = Newmark(prob, v0, beta=0.3, gamma = 0.3)
    end = 10.0
    t, y = nmark.simulate(end)
    print y[-1], f(end), t[-1]
    nose.tools.assert_almost_equal(y[-1], f(end))
