import numpy as np
import scipy as sp
from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftfreq
from scipy.fft import fftshift
import imageio
import cv2

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
# from matplotlib.animation import PillowWriter


'''
mbfnabla - mbfpartial -mitOmega mscrF mbfU
(𝛁² - 1/c² 𝛛²/𝛛t² ) 𝐔(x,y,z,t) == 0  
A(kx,ky) == ℱ(u(x,y,0)) 
u(x,y,z) == ℱ⁻¹ [A(kx,ky)exp(-iz√(k² - ky² - kx²)]       

'''

# x = np.arange(-5, 5, 0.1)
# y = np.arange(-5, 5, 0.1)
# xx, yy = np.meshgrid(x, y, sparse=True)
# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
# h = plt.contourf(x, y, z)
# plt.axis('scaled')
# plt.show()

# slit and wave wevelength
D = .1
lam = 660

# space x,y plane
g = np.linspace(-2, 2, 1600)
x, y = np.meshgrid(g, g)

# u at the slit z = 0 ?
U0 = (np.abs(x) < D/2) * (np.abs(x) < 1/2) * 1.
# print (U0.astype(float))
print(U0.shape)
plt.figure(figsize=(5, 5))
# plt.pcolormesh(x, y, U0)  # pcolor or imshow(U0) better
plt.imshow(U0)
plt.xlabel('X-Position [mm]')
plt.ylabel('Y-Position [mm]')
plt.show()

# Construct A and kx ky
A = fft2(U0)
dx = np.diff(g)[0] 
kg = fftfreq(n=len(g), d = dx ) * 2 * np.pi  # multiply by 2pi to get angular frequency
kx, ky = np.meshgrid(kg,kg)
print(A.shape,kg.shape,kx.shape)
plt.figure(figsize=(5,5))
plt.pcolormesh(fftshift(kx), fftshift(ky), np.abs(fftshift(A)))
plt.xlabel('$k_x$ [mm$^{-1}$]')
plt.ylabel('$k_y$ [mm$^{-1}$]')
plt.xlim(-100,100)
plt.ylim(-100,100)
plt.show()

Uzk = lambda z,k : ifft2(A*np.exp(-1j*z*np.sqrt(k**2-kx**2-ky**2)))

print (np.max(kx**2))
k = 1e6 *  2*np.pi / (lam)
d = 100

U = Uzk(d,k)

plt.figure(figsize=(5,5))
plt.pcolormesh(x,y,np.abs(U), cmap='inferno')
plt.xlabel('$x$ [mm]')
plt.ylabel('$y$ [mm]')
plt.show()

m  = np.arange(1,5,1)*1e-6
x_m = np.sqrt(m**2 * lam**2 * d**2 / (D**2 - m**2 * lam**2))

plt.plot(g, (np.abs(U)**2)[250])
[plt.axvline(np.abs(a), ls='--', color='r') for a in x_m]
[plt.axvline(-np.abs(a), ls='--', color='r') for a in x_m]
plt.xlabel('$x$ [mm]')
plt.ylabel('$u(x,y,z)$ [sqrt of intensity]')
plt.show()


