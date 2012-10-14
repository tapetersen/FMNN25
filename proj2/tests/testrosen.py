#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Group: Björn Lennernäs, Tobias Alex-Petersen, Johnny Sjöberg, Andy Lundberg
#  
from  __future__  import division

import scipy.optimize as so
import sys

from scipy import linspace, array, double, all
from scipy import linalg as lg
from matplotlib.pyplot import *
from scipy.optimize import rosen, rosen_der, rosen_hess
from numpy.linalg import cholesky, inv, norm, LinAlgError

from .. import chebyquad as c
from .. import proj2 as p

solr = array([1., 1.])

def test_classic_newton():
    op = p.OptimizationProblem(rosen)
    cn  = p.ClassicNewton(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)
    
def test_newton_exact_line_search():
    op = p.OptimizationProblem(rosen)
    cn  = p.NewtonExactLine(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)
    
def test_newton_inexact_line_search():
    op = p.OptimizationProblem(rosen)
    cn  = p.NewtonInexactLine(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)

def test_quasi_newton_broyden():
    op = p.OptimizationProblem(rosen)
    cn  = p.QuasiNewtonBroyden(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)
    
def test_quasi_newton_broyden_bad():
    op = p.OptimizationProblem(rosen)
    cn  = p.QuasiNewtonBroydenBad(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)
    
def test_quasi_newton_BFSG():
    op = p.OptimizationProblem(rosen)
    cn  = p.QuasiNewtonBFSG(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)
    
def test_quasi_newton_DFP():
    op = p.OptimizationProblem(rosen)
    cn  = p.QuasiNewtonDFP(op)
    guess = array([-1.,-1.])
    sol = array([1.,1.])
    assert all(abs(solr-(cn.optimize(guess))) < 1e-4)

