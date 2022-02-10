from cv2 import repeat
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal

from matplotlib import animation


class Schrodinger:
    Nx = 201
    dx = 1/(Nx-1)
    dt = 1e-6
    mu, sigma = 1/2, 1/50
    x = np.linspace(0, 1, Nx)
    # psi = np.sqrt(2)*np.exp(1j*np.pi*x).astype(complex)
    psi = np.sqrt(2)*np.sin(np.pi*x).astype(complex)
    # psi = np.exp(-(x-mu)**2/(2*sigma**2)).astype(complex)

    # V = 1e4 * np.exp(-(x-mu)**2/(2*sigma**2))
    V = -2e4 * np.exp(-(x-mu)**2/(2*sigma**2))
    psi_js = None
    E_js = None
    cs = None

    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 4))
        # ax.grid()
        self.ln2, = plt.plot([], [], 'r-', lw=2, markersize=8, label='Psi')
        self.ln1, = plt.plot([], [], 'b--', lw=3, markersize=8, label='V')
        self.ln3, = plt.plot([], [], 'g--', lw=3, markersize=8, label='M2')

        self.psi[[0, -1]] = 0  # boundaries

        self.time_text = self.ax.text(0.56, 91, '', fontsize=12,
                                      bbox=dict(facecolor='white', edgecolor='black'))
        self.meth2()

    def run(self, f):
        psi = self.psi
        # print('frame ', f)
        # ax.cla()
        psi[1:-1] = psi[1:-1] + 1j/2 * self.dt/self.dx**2 * \
            np.diff(psi, n=2) - 1j*self.dt*self.V[1:-1]*psi[1:-1]

        # normal = (np.absolute(psi)**2).sum() * self.dx
        # print('normal ' , normal )
        # psi /= normal
        # psi[[0, -1]] = 0  # boundaries

        phi = (np.absolute(psi)**2)  # *1e1

        psi2 = self.psi_m2(f*self.dt)
        # normal2 = (np.absolute(psi2)**2).sum() *self.dx
        # print('normal2 ' , normal2 )
        #psi2 /= normal2
        phi2 = np.absolute(psi2)**2  # * 1e1

        phi = phi/np.linalg.norm(phi) *1e2
        phi2 = phi2/np.linalg.norm(phi2) *1e2

        self.ln1.set_data(self.x, self.V)
        self.ln2.set_data(self.x, phi)
        self.ln3.set_data(self.x, phi2)
        self.time_text.set_text(
            '$(10^4 mL^2)^{-1}t=$'+'{:.1f} psi={:.1f}%'.format(100*f*self.dt*1e4, np.max(phi)))

        # ax.plot(self.x, self.V,
        #         color='red', alpha=0.4)
        # ax.plot(self.x, phi,
        #         color='purple', alpha=0.6)
        self.psi = psi
        # print(phi.min() , phi.max())

    def simulate(self):
        ax = self.ax
        ax.set_ylim(-1e1, 1e2)
        ax.set_xlim(0, 1)
        ax.set_ylabel('$|\psi(x)|^2$', fontsize=20)
        ax.set_xlabel('$x/L$', fontsize=20)
        ax.legend(loc='upper left')
        ax.set_title('$(mL^2)V(x) = -10^4 \cdot n(x, \mu=L/2, \sigma=L/20)$')
        ani = animation.FuncAnimation(
            self.fig, self.run, frames=5000, interval=25, repeat=False)

        plt.tight_layout()
        plt.show()

    def meth2(self):
        d = 1/self.dx**2 + self.V[1:-1]
        e = -1/(2*self.dx**2) * np.ones(len(d)-1)
        w, v = eigh_tridiagonal(d, e)
        print('shapes ', w.shape, v.T.shape)  # 201 - 2 boundaries
        self.E_js = w
        self.psi_js = np.pad(v.T, [(0, 0), (1, 1)], mode='constant')
        self.cs = np.dot(self.psi_js, self.psi.copy())

    def psi_m2(self, t):
        return self.psi_js.T@(self.cs*np.exp(-1j*self.E_js*t))


if __name__ == "__main__":
    shrodinger = Schrodinger()
    shrodinger.simulate()
