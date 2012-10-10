Results
========


Rosenbrock
----------
All methods worked on the Rosenbrock problem for sensible guesses (e.g. guess = [-1,-1]). Good Broyden was the most sensible though and variables such as the initial alpha in the stepsize finder determined whether the algorithm converged or not. 

Chebyquad
---------
All methods worked for problems of size 4.

For problems of size 8 all quasi newton methods are successfull but the newton methods all fail beacuse the hessian isn't positive definite. 

Similar results were obtained for problems of size 11. 


Task 12
-------
For the Rosenbrock problem the approximated inverse and the inverse of the numerical inverse stay close to each other for all k. For the chebyquad problem however the difference between them grow very large when the algorithm is close to terminating, ie. for high k. The difference between them tend to fluctuate as well for k not close to the end. The numerical inverse of the hessian grows very large when close to terminating while the approximated inverse is populated by much smaller values. It works the other way around to, trying to take the inverse of the approximated inverse results in a matrix with very large value while the numerical hessian has fairly small values. A reason for this may be that the computation of the inverse is particularly ill conditioned near the minimum, but we have not found any good reason for why this may be the case. 
