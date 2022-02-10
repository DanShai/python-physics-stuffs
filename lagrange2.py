import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp, odeint


t, g, H, M, phi = sm.symbols('t g H M \phi', real=True, positive=True)
r = sm.symbols('r', cls=sm.Function)
r = r(t)
drdt = sm.diff(r, t)

T = sm.Rational(1, 2) * M * drdt**2
U = -1 * M * g * r * sm.sin(phi)
L = T - U
LE = sm.diff(L, r) - sm.diff(sm.diff(L, drdt), t)

solm = sm.solve(LE, sm.diff(r, t, t))
d2rdt = solm[0]

# S = (r,v)
f_drdt = sm.lambdify(drdt, drdt)
f_dvdt = sm.lambdify((t, g, phi, M, H, r, drdt), d2rdt)


def dSdt(t, S, g, phi, M, H):
    # print(S)
    r, v = S
    return [
        f_drdt(v),
        f_dvdt(t, g, phi, M, H, r, drdt)
    ]


t = np.linspace(0, 10, 101)  # s
g = 9.81  # m/s^2
M = 5  # kg
phi = np.pi/4
H = 100.
S0 = [0, 0]


# sol = odeint(dSdt, y0=S0, t=t, args=(g, phi, M),tfirst=True)
sol = solve_ivp(dSdt, t_span=(0, max(t)), y0=S0, t_eval=t, args=(g, phi, M, H))
r_sol = sol.y[0]
v_sol = sol.y[1]

plt.plot(t, r_sol)
plt.plot(t, v_sol)


def XY(r): return (r * np.cos(phi), H - r * np.sin(phi))


x, y = XY(r_sol)

# print (np.where(y <= 0))
y = y[:64]
x = x[:64]


def animate(i):
    ln1.set_data([0, x[i]], [0, y[i]])
    ln2.set_data([0, x[0], x[-1], 0], [0, y[0], y[-1], y[-1]])


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.set_facecolor('k')
# https://stackoverflow.com/questions/14088687/how-to-change-plot-background-color
ax.set_facecolor('xkcd:sky blue')
# ax.set_facecolor((1.0, 0.47, 0.42))
# ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
# ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks
ln1, = plt.plot([], [], 'o--', color='orange', lw=3, markersize=16)
ln2, = plt.plot([], [], 'ro--', lw=3, markersize=8)
xl, yl = np.abs(x).max() + 10, np.abs(y).max() + 10
print(xl, yl)
ax.set_xlim(-10, xl)
ax.set_ylim(-yl, yl)
plt.tight_layout()
ani = animation.FuncAnimation(
    fig, animate, frames=64, interval=32, repeat=True)
# ani.save('inclined.gif', writer='pillow', fps=25)

plt.show()
