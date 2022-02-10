import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

'''
y₁′  = y₁ + y₂² + 3x 
y₂′  = 3y₁ + y₂² - cos(x)   

define S = (y1,y2)
'''


def dSdx(x, S):
    y1, y2 = S
    return (y1 + y2**2 + 3*x, 3*y1 + y2**3 - np.cos(x))

y1_0 = y2_0 = 0
S_0 = (y1_0,y2_0)

## solve

x = np.linspace(0,1,100)
sol = odeint(dSdx,y0=S_0,t=x , tfirst=True)
sol_2 = solve_ivp(dSdx, t_span=(0,max(x)), y0=S_0, t_eval=x)

y1_sol = sol[:,0]
y2_sol = sol[:,1]
print(sol.shape , sol_2.y.shape)
y1_sol2 = sol_2.y[0,:]
y2_sol2 = sol_2.y[1,:]

print (np.allclose(y1_sol,y1_sol2 ,rtol=1e-2) )
plt.plot(x, y1_sol,'--')
plt.plot(x, y2_sol,'--')
plt.plot(x, y1_sol2)
plt.plot(x, y2_sol2)

'''
Python does not have functions to directly solve second order ODEs 
But **any second order ODE can be converted into two first order ODEs

ẍ  = -ẋ²   + sin(x)
We can convert this into two first order ODEs as follows:

* Take x (this is what we're trying to solve for). Then define ẋ =v so that v becomes a new variable.
* Note that ẋ = v is one differential equation
* Since v̇  = ẍ  we get another differential equation

Our two equations:

ẋ  = v
v̇  = -v² + sin(x)
These are two coupled first order equations. They require an initial condition (x₀ , v₀ )

'''

def dSdt(x, S):
    x, v = S
    return [v,
           -v**2 + np.sin(x)]
x_0 = 0
v_0 = 5
S_0 = (x_0, v_0)

t = np.linspace(0, 1, 100)
# sol = odeint(dSdt, y0=S_0, t=t, tfirst=True)
# x_sol = sol[:,0]
# v_sol = sol[:,1]

sol = solve_ivp(dSdt, t_span=(0,max(t)), y0=S_0, t_eval=t)
x_sol = sol.y[0,:]
v_sol = sol.y[1,:]

plt.plot(t, x_sol)
plt.plot(t, v_sol)

plt.show()

