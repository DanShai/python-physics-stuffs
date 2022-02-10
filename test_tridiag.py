
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal



d = 3*np.ones(4)
print(d)

e = -1*np.ones(3)
print(e)

w, v = eigh_tridiagonal(d, e)
print(w)
print(v)


A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
print(A)

isSame = np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
print(isSame)