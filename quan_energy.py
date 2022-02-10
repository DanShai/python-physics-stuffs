import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

N = 100  #20
dx = 1/N
x = np.linspace(0, 1, N+1)
Vx = 1000*np.sin(20*x) * x**4 

# important! derivative
d = (1/dx)**2 + Vx[1:-1]
e = -1/(2*dx**2) * np.ones(len(d)-1)


A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
w1, psi1 = np.linalg.eigh(A)
w, psi = eigh_tridiagonal(d, e)

print(w[:10])
print(w1[:10])

print(psi.T[:2 , :10])
print(psi1.T[:2 , :10])

fig, ax = plt.subplots()



ax.plot(x[1:-1], psi.T[0]**2)
ax.plot(x[1:-1], psi.T[1]**2)
ax.plot(x[1:-1], psi.T[2]**2)
ax.plot(x[1:-1], psi.T[3]**2)
# ax.plot(x, Vx )



# plt.bar(np.arange(0, 10, 1), w[0:10])
# plt.ylabel('$mL^2 E/\hbar^2$')

plt.show()
