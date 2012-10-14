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


Task 5
-------
Using the built in 1d minimizer fmin_bound newtons problem follows the valley of
the rosenbrock function and solves it in 12 iterations starting from (-1, 1).
This is actually worse than the pure newton that makes a jump to (1, -3) and
from there finds the minimum directly.

Task 7
-------
This test is performed in tests/testmisc.py as test_linesearch() and finds the
same alpha as the book

Task 9
-------
These are all in the proj2.py. The bad broyden is implemented as on wikipedias
article on Broyden's method but it performs very poor and the one they mark as
good is not the same as the one in the lecture notes.

Task 10
-------
We changed the functions to use scipys built in versions of the chebyshev
polynomials for a quite significant speed boost. The optimizations find the same
minimum as before but with fewer iterations propbaly depending on better
numerical stability, or perhaps pure luck.

Task 11
-------
These minimizations are done in tests/testquasi.py and tests/testnewton.py for
4, 8 and 11. All of the QuasiNewton variants except BroydenBad succeed finding a minimum as good as the
scipy version but sometimes the parameters are permuted, is the problem maybe
symmetric? The normal newton variants with linesearches fails in some cases on
dimension 8 and 11 and take very long time.

Task 12
-------

For the Rosenbrock problem the approximated inverse and the inverse of the
numerical inverse stay close to each other for all k. For the chebyquad problem
however the difference between them grow very large when the algorithm is close
to terminating, ie. for high k. The difference between them tend to fluctuate as
well for k not close to the end. 

The numerical inverse of the hessian grows very large when close to terminating
while the approximated inverse is populated by much smaller values. It works the
other way around too, trying to take the inverse of the approximated inverse
results in a matrix with very large value while the numerical hessian has fairly
small values. A reason for this may be that the computation of the inverse is
particularly ill conditioned near the minimum, but we have not found any good
reason for why this may be the case. Is it even supposed to converge at the
minimum for non-concave functions?
