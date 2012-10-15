
from assimulo import testattr
from assimulo.explicit_ode import *
from assimulo.problem import Explicit_Problem, cExplicit_Problem
from assimulo.exception import *
import numpy as N
import pylab as P
from assimulo.solvers import CVode
import nose
from scipy import double
import newmark as ne

def test_newmark_basic_2nd_order():
	f  = lambda x: 9.82/2.0 * t**2  + 5
	""" Take off in a rocket at g-acc from 5m above grounds """
	f2 = lambda x, y: 9.82
	y0 = 5.0
	v0 = 0.0
	prob = Explicit_Problem(f2, y0)
	nmark = Newmark(f2, v0)
	end = 10
	t, y = nmark.simulate(0,end)
    nose.tools.assert_almost_equal(y, f(end))

