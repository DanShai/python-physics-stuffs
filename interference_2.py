from cv2 import repeat
import numpy as np
import scipy as sp
from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftfreq
from scipy.fft import fftshift
from scipy.stats import multivariate_normal
# import imageio
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
# from matplotlib.animation import PillowWriter
from matplotlib import cm

'''

https://www.youtube.com/watch?v=594t2HEcWo0&list=PLQms29D1RqeJFxsVHyploQELPJOUn8OSw&index=3
https://www.youtube.com/watch?v=QeNHF-H_ANM
mbfnabla - mbfpartial -mitOmega mscrF mbfU
(𝛁² - 1/c² 𝛛²/𝛛t² ) 𝐔(x,y,z,t) == 0  
A(kx,ky) == ℱ(u(x,y,0)) 
u(x,y,z) == ℱ⁻¹ [A(kx,ky)exp(-iz√(k² - ky² - kx²)]       

'''

S = .2
D = .05
Hy = 2
# space x,y plane
sz = 500
lam = 620
fr = 40
g = np.linspace(-4, 4 , sz)
x, y = np.meshgrid(g, g)

# cmap = LinearSegmentedColormap.from_list('custom', 
#                                          [(0,0,0),wavelength_to_rgb(lam)],
#                                          N=256) 
cmap = 'afmhot'


def gauss(X, Y, mu=[], sigma=[]):
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty((2,) + X.shape)
    pos[0, :, :] = X
    pos[1, :, :] = Y
    print('pos ', pos.T.shape)

    F = multivariate_normal(mu, sigma)
    Z = F.pdf(pos.T).T
    return Z

def gauss2(X, Y, mu, sigma):

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    print('pos ', pos.shape)
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    n = mu.shape[0]
    Sigma_det = np.linalg.det(sigma)
    Sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    #fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    # https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    fac = np.einsum('ijk,kl,ijl->ij', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N


# def slit_example():
#     just_example= np.select([((np.abs(x-S/2) < D/2) + (np.abs(x+S/2) < D/2)) * (np.abs(y) < 2),
#                       ((np.abs(x-S/2) >= D/2) + (np.abs(x+S/2) >= D/2)) * (np.abs(y) >= 2)],
#                      [1.,
#                       0])
#     plt.figure(figsize=(5, 5))
#     plt.contourf(x,y,just_example, cmap='hot')
#     plt.xlabel('X-Position [mm]')
#     plt.ylabel('Y-Position [mm]')
#     plt.show()

# slit_example()

def makeU0(u0='gaussian'):
    if u0 == 'gaussian':
        # u at the slit z = 0  square func will give sinc transform
        U01 = ((np.abs(x-S/2) < D/2) + (np.abs(x+S/2) < D/2)) * (np.abs(y) < Hy) * 1.
        #U0 = ((x-S/2)**2 + (y)**2 < (D/2)**2) + ((x+S/2)**2 + (y)**2 < (D/2)**2) * 1.

        a ,b = 2*D , Hy/4
        mu1 = np.array( ( -S/2 - D/2 ,0) )
        mu2 = np.array( ( S/2 + D/2 ,0) )
        sigma = np.array([[a, a*b], [-a*b,  b]])
        print('det ' , np.linalg.det(sigma))
        U02  =  (gauss2(x, y, mu=mu1, sigma=sigma) + gauss2(x, y, mu2, sigma=sigma) ) 
        # U0 = np.where(U01,U02,0) 
        # print (U0.astype(float))
        # plane waves mixed with gaussians
        U0 = U01  * U02 * 5e-1
        print(U0.shape)
    elif u0=='hexa':
        U0 = U0_from_img(im='hexagon_grating.jpg')
        print( U0.shape)
        U0 = U0.sum(axis=2).astype(float)
        print( U0.shape)
    return U0


def surf_plot(u=None,title='U'):
    fig = plt.figure(figsize=(5, 5))
    mmap = plt.get_cmap('hot')
    ax = fig.add_subplot(projection = '3d')
    ax.plot_surface(x, y, u, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=mmap)
    cset = ax.contourf(x, y, u, zdir='z', offset=-0.15, cmap=cm.viridis)
    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,.5)
    ax.set_zticks(np.linspace(0,0.1,1))
    # ax.view_init(27, -21)    
    ax.view_init(elev=10, azim=-45)
    ax.set_xlabel('X-Position [mm]')
    ax.set_ylabel('Y-Position [mm]')
    ax.set_zlabel('U0 [u]')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# https://www.youtube.com/watch?v=594t2HEcWo0&list=PLQms29D1RqeJFxsVHyploQELPJOUn8OSw&index=3
# aperture func
def drawA(u0=None):
    # Construct A and kx ky
    A = fft2(u0)
    dx = np.diff(g)[0] 
    kg = fftfreq(n=len(g), d = dx ) * 2 * np.pi  # multiply by 2pi to get angular frequency
    kx, ky = np.meshgrid(kg,kg)
    print(A.shape,kg.shape,kx.shape)
    plt.figure(figsize=(5,5))
    plt.pcolormesh(fftshift(kx), fftshift(ky), np.abs(fftshift(A)))
    # plt.imshow(np.abs(fftshift(A)), cmap='hot')
    plt.xlabel('$k_x$ [mm$^{-1}$]')
    plt.ylabel('$k_y$ [mm$^{-1}$]')
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.title(' A ( cross-section )')
    plt.tight_layout()
    plt.show()    


def Uzk(u0=None,z=50, lam=lam):
    A = fft2(U0)
    dx = np.diff(g)[0]
    kg = 2*np.pi * fftfreq(n=len(g), d=dx)
    kx, ky = np.meshgrid(kg, kg)
    k = 1e6 * 2*np.pi/lam
    return np.abs(ifft2(A*np.exp(-1j*z*np.sqrt(k**2-kx**2-ky**2))) ) 


def show_U(u=None,title='U'):
    plt.figure(figsize=(5, 5))
    #plt.pcolormesh(x,y,np.abs(U), cmap='plasma')
    # plt.imshow(np.abs(U)**2, cmap=cmap)
    plt.contourf(x,y, u, cmap=cmap)
    plt.xlabel('$x$ [mm]')
    plt.ylabel('$y$ [mm]')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def show_m(z=1e2):
    m  = np.arange(1,5,1)* 1e-6
    DS = (S + D )
    #x_m = np.sqrt(m**2 * lam**2 * z**2 / (DS**2 - m**2 * lam**2))
    x_m = m * lam * z / DS
    plt.figure(figsize=(5, 5))
    plt.plot(g, U[sz//2])
    [plt.axvline(np.abs(a), ls='--', color='r') for a in x_m]
    [plt.axvline(-np.abs(a), ls='--', color='r') for a in x_m]
    plt.xlabel('$x$ [mm]')
    plt.ylabel('$u(x,y,z)$ [sqrt of intensity]')
    plt.title(' m nodes of U ')
    plt.tight_layout()
    plt.show()


# cros section
# central_line = (np.abs(U) ** 2)[sz//2]
# plt.plot(g, central_line)
# plt.xlabel('$x$ [mm]')
# plt.ylabel('$u(x,y,z)$ [sqrt of intensity]')
# plt.grid()
# plt.show()



def U0_from_img(im='hexagon.jpg'):
    img = cv2.imread(im)
    img = np.pad(img, 25, mode='constant')
    img = cv2.resize(img, dsize=(sz, sz), interpolation=cv2.INTER_CUBIC)
    return np.array(img)




def run(z0=2e2):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.set_xlabel('X-Position [mm]')
    ax.set_ylabel('Y-Position [mm]')
    time_text = ax.text(0.5, 0.95, '', fontsize=10,
                        bbox=dict(facecolor='white', edgecolor='black'), 
                        transform=ax.transAxes)
    ax.set_title(' Uz ')
    plt.tight_layout()
    
    def animate(f):
        time_text.set_text('f={:.2f} z={:.2f}'.format((f+1),(f+1)/fr*z0))
        U = Uzk(z=(f+1)/fr * z0 , lam=lam)
        # ax.imshow(np.abs(U), cmap=cmap)
        #ax.pcolormesh(x,y,np.abs(U), cmap=cmap, vmax=np.max(np.abs(U))/2)
        ax.contourf(x,y,U, cmap=cmap)
        
        
    ani = animation.FuncAnimation(fig, animate, frames=fr, interval=fr//2, repeat=False )
    plt.show()


def run3D(z0=2e2):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    mmap = plt.get_cmap('hot')
    ax = fig.add_subplot(projection = '3d')
    time_text = ax.text(0.5, 0.95,1, '', fontsize=10,
                        bbox=dict(facecolor='white', edgecolor='black'), 
                        transform=ax.transAxes)
    ax.set_title(' Uz ')
    ax.set_zlim(-0.15,.5)
    ax.set_zticks(np.linspace(0,0.1,1))
    # ax.view_init(27, -21)    
    ax.view_init(elev=10, azim=-45)
    ax.set_xlabel('X-Position [mm]')
    ax.set_ylabel('Y-Position [mm]')
    ax.set_zlabel('U [u]')
    ax.set_title('Uz')
    
    def animate(f):
        ax.cla()
        time_text.set_text('f={:.2f} z={:.2f}'.format((f+1),(f+1)/fr*z0))
        U = Uzk(z=(f+1)/fr * z0 , lam=lam)
        ax.plot_surface(x, y, U, rstride=3, cstride=3, linewidth=1, antialiased=True,
                        cmap=mmap)
        cset = ax.contourf(x, y, U, zdir='z', offset=-0.15, cmap=cm.viridis)

        
    ani = animation.FuncAnimation(fig, animate, frames=fr, interval=fr//2, repeat=False )
    plt.tight_layout()    
    plt.show()




U0 = makeU0(u0='gaussian')
show_U(u=U0,title='U0')
surf_plot(u=U0,title='U0')
drawA(u0=U0)
U = Uzk(u0=U0,z=1e2, lam=lam)
show_U(u=U,title='U')
surf_plot(u=U,title='U')
show_m()

U0 = makeU0(u0='hexa')
U = Uzk(u0=U0,z=100, lam=lam)
# run(z0=.8e3)
run3D(z0=.8e3)

