Overview
========
Our solution can be conceptually divided into two main parts: solvers and help
methods, where the help methods are used by the solvers. 

Testing is also perfomed using nosetests. 


Help methods
------------
The solvers need access to the function and the gradiant and hessian of the
funtion. We've bundled together this functionality in a problem class which
provides the aforementioned functions. 

Other help methods include a method that interpolates a minimum using cubic or
quadratic polynomials fitted to the derivatives and values at end points. Used
for linesearch.


Solvers
--------
The solvers are written using inheritance from AbstractNewton. The optimize
contains all logic and the subclasses speialize it by overriding the derivation
of the hessian, the way to get a search direction and the linesearch for a
minimum along given direction.


Testing
--------
Tests include testing of gradient and hessian computation, the minimization
function, solvers and linesearch. The tests for the solvers are considered
successfull if the solver terminates and the function value of the obtained
solution is very close to the function value of the obtained solution for
scipy:s optimization method. 

