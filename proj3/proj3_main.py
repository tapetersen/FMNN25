from scipy       import *
from matplotlib.pyplot import *
from numpy import min, max
from numpy.linalg import norm,solve
import problem_type as p

def run_example(with_plots=True):
    
    #Define the rhs
    def f(t,y):
        ydot = -y[0]
        return N.array([ydot])
    
    #Define an Assimulo problem
    exp_mod = Newmark(f, y0=4.0)
    exp_mod.name = 'Simple CVode Example'
    
    #Define an explicit solver
    exp_sim = CVode(exp_mod) #Create a CVode solver
    
    #Sets the parameters
    exp_sim.iter  = 'Newton' #Default 'FixedPoint'
    exp_sim.discr = 'BDF' #Default 'Adams'
    exp_sim.atol = [1e-4] #Default 1e-6
    exp_sim.rtol = 1e-4 #Default 1e-6
    
    #Simulate
    t1, y1 = exp_sim.simulate(5,100) #Simulate 5 seconds
    t2, y2 = exp_sim.simulate(7) #Simulate 2 seconds more
    
    #Basic test
    #nose.tools.assert_almost_equal(y2[-1], 0.00347746, 5)
    
    #Plot
    if with_plots:
        P.plot(t1, y1, color="b")
        P.plot(t2, y2, color="r")
        P.show()


def main():
    run_example()

if __name__ == '__main__':
    main()
