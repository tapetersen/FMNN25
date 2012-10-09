Project 2
============================


Help classes and methods
-------------------------

.. autoclass:: proj2.FunctionTransforms
   :members:
   
   .. automethod:: proj2.FunctionTransforms.__init__
   
   .. automethod:: proj2.FunctionTransforms.__call__
  
.. autoclass:: proj2.OptimizationProblem
   :members:
   
   .. automethod:: proj2.OptimizationProblem.__init__
   
   .. automethod:: proj2.OptimizationProblem.__call__
   
.. autofunction:: proj2.cubic_minimize
.. autofunction:: proj2.find_step_size


Solvers
-------------


.. autoclass:: proj2.AbstractNewton
   :members:
   :undoc-members:
   
   .. automethod:: proj2.AbstractNewton.__init__
   
Newton solvers
~~~~~~~~~~~~~~~
.. autoclass:: proj2.ClassicNewton
   :members:
   
.. autoclass:: proj2.NewtonExactLine
   :members:
   
.. autoclass:: proj2.NewtonInexactLine
   :members:



Quasi-Newton solvers
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proj2.QuasiNewtonBroyden
   :members:

.. autoclass:: proj2.QuasiNewtonBroydenBad
   :members:
   
.. autoclass:: proj2.QuasiNewtonBFSG
   :members:
   
.. autoclass:: proj2.QuasiNewtonDFP
   :members:











