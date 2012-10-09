Overview
========

Our solution can be conceptually divided into two main parts: solvers and help methods, where the help methods are used by the solvers. 

Testing is also perfomed using nosetests. 


Help methods
------------
The solvers need access to the function and the gradiant and hessian of the funtion. We've bundled together this functionality in a problem class which provides the aforementioned functions. 

Other help methods include a method that can be used to find the minimum of an interpolation of provided data, this is used when calculating step size in some solvers. 

Solvers
--------
The solvers are written using inheritance where the Classic Newton method provides the most useful abstraction. It is written on a form that calls other methods to update the hessian, to find the step direction, and to find the step size. In this way other solvers, such as exact/inexact line search and quasi-newton method, can be easily implemented by subclassing overriding these functions.



Testing
--------
Tests include testing of gradient and hessian computation, the minimization function, solvers and linesearch. The tests for the solvers are considered successfull if the solver terminates and the function value of the obtained solution is very close to the function value of the obtained solution for scipy:s optimization method. 
