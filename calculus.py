import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = np.linspace(0, 10, 1000)
y = np.exp(-x/10)*np.sin(x)
dydx = np.gradient(y,x)
dx = x[1]-x[0]
y_intg = np.cumsum(y) * dx

cond = (x >= 4) & (x <= 7)
# cond = (x >= 4)*(x <= 7) # same
x1 = x[cond]
y1 = y[cond]
print(y1.mean(), y1.std())

# inflextion_points
in_p = x[1:][( (dydx[1:]*dydx[:-1]) < 0) ]
print(in_p)
idx = np.array([np.where( np.isclose(x,in_p[0],1e-6))[0] ,
np.where( np.isclose(x,in_p[1],1e-6))[0] ,
np.where( np.isclose(x,in_p[2],1e-6))[0] ])
idx = idx.ravel()
print (idx)
print( y[ idx] )
ax.plot(x, y,  color='red', alpha=0.4)
ax.plot(x1, y1, color='blue', alpha=0.4)
ax.plot(x, y_intg, color='purple', alpha=0.4)
ax.scatter(in_p, y[idx],
                   s=10, c='pink', edgecolor='red')

plt.tight_layout()
plt.show()
