Overview
======== 
We implemented the methods by creating two classes, Newmark and HHT, which inherits from Explicit_ODE. Here the init, _step and intergrate methods are overridden. The solvers store additional necessary variables such as ybis and yprime and the beta and gamma variables. Each step updates y using the corresponding methods. 

This project is the Handin of Björn Lennernäs and Tobias Alex-Petersen. 

_step
------
For the explicit form of the Newmark method a step is fairly straightforward, but for the implicit form of Newmark and HHT we need to solve a system in each step. For these methods the active, unknown, variable is a_{n+1} for each variable. Thus we solve for a_{n+1} by creating a residual equation, as in this snippet from the Newmark method::
    # The active variable is a_n1
    
    f_pn1 = lambda a_n1:  (y + h*self.v + (h**2 / 2.0) * \
                               ((1.0 - 2.*self.beta)*self.a + 2.*self.beta*a_n1))
                               
    f_vn1 = lambda a_n1:  (self.v + h *((1.0-self.gamma)*self.a + self.gamma*a_n1))
    
    f_an1 = lambda a_n1: a_n1 - (self.f(f_pn1(a_n1),f_vn1(a_n1),t+h))
                
    a = fsolve(f_an1, self.a)  # Solve w.r.t a_n1
                
    y      = f_pn1(a) # Calculate and store new variables. 
    
    self.v = f_vn1(a)
    
    self.a = a
    
    return t+h, y

