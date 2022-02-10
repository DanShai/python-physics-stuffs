from cv2 import repeat
import numpy as np
import sympy as sm
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint
from scipy.integrate import solve_ivp



# step 0
# ------------------------  Define symbols ------------------------------
t, g, l1, l2, m1, m2, m3, k, L0 = sm.symbols('t g l_1 l_2 m_1 m_2 m_3 k L_0')
theta1, theta2 = sm.symbols('theta_1 theta_2', cls=sm.Function)

theta1 = theta1(t)
theta2 = theta2(t)
dtheta1 = sm.diff(theta1, t)
dtheta2 = sm.diff(theta2, t)
d2theta1 = sm.diff(dtheta1, t)
d2theta2 = sm.diff(dtheta2, t)

x1, y1 = l1*sm.cos(theta1), -l1*sm.sin(theta1)
x2, y2 = 2*x1, 0.
x3, y3 = x2 + l2*sm.sin(theta2), -l2*sm.cos(theta2)

# step 1
# ------------------------  Define T U L ------------------------------
sm.pprint(x1)
T = sm.Rational(1,2) * m1 * (sm.diff(x1,t)**2 + sm.diff(y1,t)**2) \
    +sm.Rational(1,2) * m2 * (sm.diff(x2,t)**2 + sm.diff(y2,t)**2) \
    +sm.Rational(1,2) * m3 * (sm.diff(x3,t)**2 + sm.diff(y3,t)**2)

V = m1*g*y1 + m2*g*y2 + m3*g*y3 +sm.Rational(1,2) * k * (x2-L0)**2
L =T-V

# sm.pprint(L)
# step 2
#--------------------- Solve Lagrangian on the accelerations ---------------------
LE1 = sm.diff(L, theta1) - sm.diff(sm.diff(L, dtheta1), t).simplify()
LE2 = sm.diff(L, theta2) - sm.diff(sm.diff(L, dtheta2), t).simplify()
sols = sm.solve([LE1, LE2], (d2theta1, d2theta2),
                simplify=False, rational=False)

# sm.pprint(sols)

# step 3 : 
# --------------- reconstract first degree odes from the second degree θ̈ ---------------ᵢ 
# θ̇ = ω  and ω̇ = θ̈ 
# use numeric for the odes
# sm.lambdify(vars, equation/expression)
f_dtheta1 = sm.lambdify(dtheta1, dtheta1)
f_dw1dt = sm.lambdify((t,g,k,L0,m1,m2,m3,l1,l2,theta1,theta2,dtheta1,dtheta2), sols[d2theta1])
f_dtheta2 = sm.lambdify(dtheta2, dtheta2)
f_dw2dt = sm.lambdify((t,g,k,L0,m1,m2,m3,l1,l2,theta1,theta2,dtheta1,dtheta2), sols[d2theta2])

# step 4 
# ---------------- define S⃗ = (θᵢ , ωᵢ ) =( θᵢ  , θ̇ᵢ  ) i = 1,2 -----------------------------------------
#  f_dw1dt should be called f_d2theta1 but we follow 1st order equation rules
#  [dS/dt] = (dy1/dt , dy2/dt)  = [rhs expressions vecctor]

def dSdt(t, S, g, k, L0, m1, m2, m3, l1, l2):
    # print(S)
    theta1, w1, theta2, w2 = S
    return [
        f_dtheta1(w1),
        f_dw1dt(t,g,k,L0,m1,m2,m3,l1,l2,theta1,theta2,w1,w2),
        f_dtheta2(w2),
        f_dw2dt(t,g,k,L0,m1,m2,m3,l1,l2,theta1,theta2,w1,w2),
    ]

# step 5 
# ------------------- solve ode numericaly and plot -------------------------------------------------

t = np.linspace(0, 40, 1001) # s
g = 9.81 #m/s^2
k = 30 # N/m
m1=2 # kg
m2=2 # kg
m3=1 # kg
l1 = 1 # m
l2 = 1 # m
L0 = 1.5*l1 # m
S_0 = [1, 2, 2, 0]
# sol = odeint(dSdt, y0=S_0, t=t, args=(g, k, L0, m1, m2, m3, l1, l2),tfirst=True)
# theta1_sol = sol[:,0]
# omega1_sol = sol[:,1]
# theta2_sol = sol[:,2]
# omega2_sol = sol[:,3]

sol = solve_ivp(dSdt, t_span=(0,max(t)), y0=S_0, t_eval=t,args=(g, k, L0, m1, m2, m3, l1, l2))
theta1_sol = sol.y[0]
omega1_sol = sol.y[1]
theta2_sol = sol.y[2]
omega2_sol = sol.y[3]

plt.plot(t, theta1_sol)
plt.plot(t, theta2_sol)

def get_x1y1x2y2x3y3(t, the1, the2, l1, l2):
    return (l1*np.cos(the1),
            -l1*np.sin(the1),
            2*l1*np.cos(the1),
            np.zeros(len(the1)),
            2*l1*np.cos(the1) + l2*np.sin(the2),
            -l2*np.cos(the2))

x1, y1, x2, y2, x3, y3 = get_x1y1x2y2x3y3(t, theta1_sol, theta2_sol, l1, l2)

def animate(i):
    ln1.set_data([0, x1[i], x2[i], x3[i]], [0, y1[i], y2[i], y3[i]])
    ln2.set_data([0, x2[i]], [0, y2[i]])
    
fig, ax = plt.subplots(1,1, figsize=(5,5))
# ax.set_facecolor('k')
ax.set_facecolor('xkcd:sky blue')
ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks
ln1, = plt.plot([], [], 'ro--', color='purple', lw=3, markersize=8)
ln2, = plt.plot([], [], 'ro--', lw=3, markersize=8)
ax.set_ylim(-4,4)
ax.set_xlim(-4,4)
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50,repeat=False)
ani.save('pendule.gif',writer='pillow',fps=25)


plt.show()