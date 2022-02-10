import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse

'''
Tensors === cartesian product for sets , bilinear: means its NOT linear! 

----- bilinear ------ Tensors -----    !==   ---- a double linear map ----
f(v1 + v2, w) = f(v1,w) + f(v2,w)       |   
f(v, w1 + w2) = f(v, w1) + f(v, w2)     | f(v1+v2, w1+w2) = f(v1, w1) + f(v2, w2)
f(a*v, w) = f(v, a*w) = a*f(v,w)        | f(a*v, a*w) = a*f(v, w)
-------------------------------------------------------------------------------------

It's a little like a process-state duality. On the one hand, a matrix v⊗w is a process 
it's a concrete representation of a (linear) transformation. On the other hand, v⊗w is, 
abstractly speaking, a vector. And a vector is the mathematical gadget that physicists 
use to describe the state of a quantum system. So matrices encode processes; 
vectors encode states. The upshot is that a vector in a tensor product V⊗W can be viewed
in either way simply by reshaping the numbers as a list or as a rectangle.
'''

N = 100
XY = np.linspace(0, 1, N, dtype=float)
X, Y = np.meshgrid(XY, XY)

'''
 $$\left[-\frac{1}{2}(D \oplus D) + m\Delta x^2 V \right] \psi = \left(m \Delta x^2 E\right) \psi$$

'''

# V = np.exp(-(X-0.3)**2/(2*0.1**2))*np.exp(-(Y-0.3)**2/(2*0.1**2))
# V = 1./(1+np.exp(-(X-0.3)**2/(2*0.1**2))*np.exp(-(Y-0.3)**2/(2*0.1**2)))
# V = 1./(1+(X-0.3)**2 + (Y-0.3)**2)
V = (X-0.3)**2 + (Y-0.3)**2
V = 0*V
ones = np.ones(N)
data = np.array([ones, -2*ones, ones])
diags = np.array([-1, 0, 1])
D = sparse.spdiags(data, diags, N, N)
T = -1/2 * sparse.kronsum(D, D)
U = sparse.diags(V.reshape(N**2), (0))
H = T+U

print(data.shape)
print(D.toarray().shape)
print(T.toarray().shape)
print(U.toarray().shape)
print(H.toarray().shape)

E, Psi = eigsh(H, k=10, which='SM')
print(E.shape, Psi.shape)

def Psi_n(n): return Psi.T[n].reshape((N, N))


plt.figure(figsize=(5,5))
plt.subplot(211)
plt.contourf(X, Y, Psi_n(4)**2, 20)




alpha = E[0]/2
E_div_alpha = E/alpha
x = np.arange(0, len(E), 1)

plt.subplot(212)
plt.scatter(x, E_div_alpha)
[plt.axhline(nx**2 + ny**2,color='r') for nx in range(1,5) for ny in range(1,5)]

plt.show()

class anima:
    my_cmap = plt.get_cmap('cool')
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    #ax = Axes3D(fig)
    n = 0
    
    def __init__(self,X,Y) -> None:
        self.X,self.Y = X,Y
        self.psi = Psi_n(0)

    def init(self):
        # Plot the surface.
        self.ax.set_xlabel('$x/a$')
        self.ax.set_ylabel('$y/a$')
        self.ax.set_zlabel('$\propto|\psi|^2$')



    def animate(self,f):
        if f % 10 == 0:
            self.n = (self.n+1)%10
        plt.cla()
        self.psi = Psi_n(self.n)
        self.ax.view_init(elev=10, azim=f)
        self.ax.plot_surface(self.X, self.Y, self.psi**2, cmap=self.my_cmap,
                        linewidth=0, antialiased=False)

    def show_anim(self):
        ani = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                    frames=501, interval=50, repeat=False)

        # ani.save('shrodinger2D.gif',writer='pillow',fps=25)
        plt.show()


ani = anima(X,Y)
ani.show_anim()