import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib import animation, cm


class Heat:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.x = np.arange(-2, 2, 0.1)

        self.dx2 = .01
        self.dt = .001
        self.a2 = 5

        #self.ux = 100 * (np.exp(-(self.x-1)**2) + np.exp(-(self.x+1)**2))
        self.ux = stats.uniform.rvs(loc=0, scale=100, size=self.x.size)
        self.ux[[0, -1]] = 20  # boundaries
        self.t_mean = []
        # self.ax.plot(self.x, self.ux,  'k',
        #              color='g', alpha=0.6)
        self.init()

    def init(self):
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(0, 100)
        #plt.autoscale(enable=False, axis='y')

    def run(self, f):
        ux, ax, tm = self.ux, self.ax, self.t_mean
        m = ux.mean()
        # tm.append(m)
        print('frame ', f, 'mean temp ', m, 'max temp ', ux.max())
        # print('ux ', self.ux.shape)
        ax.cla()
        self.init()
        c = (self.a2*self.dt/self.dx2)
        # u = ux
        # v = (u[:-2] - 2 * u[1:-1] + u[2:])
        v = c*np.diff(ux, n=2)
        ux[1:-1] = ux[1:-1] + c * v  # [0,-1] boundary consditions
        ax.plot(self.x, ux,  'k',
                color='purple', alpha=0.6)
        # ax.plot(np.arange(len(tm)), tm,  'k',
        #         color='g', alpha=0.6)
        # a = (y2 - y1) / (x2 - x1)
        # b = y1 - a * x1
        # y = a*x + b = a*x + y1 - a*x1 = a*(x-x1) + y1
        p = 16
        a = (ux[p+1] - ux[p]) / (self.x[p+1]-self.x[p])
        #a = (np.diff(ux, 1)[10])/(np.diff(self.x, 1)[10])
        b = ux[p] - a*self.x[p]
        x = self.x[p-10:p+10]

        y = a*x + b
        ax.plot(x, y,  'k',
                color='red', alpha=0.4)

        ax.scatter([x[0], x[-1]], [y[0], y[-1]],
                   s=10, c='pink', edgecolor='red')

        plt.tight_layout()

    def simulate(self):
        ani = animation.FuncAnimation(
            self.fig, self.run, init_func=self.init, frames=10000, interval=10, repeat=False)
        plt.show()


if __name__ == "__main__":
    heat = Heat()
    heat.simulate()
