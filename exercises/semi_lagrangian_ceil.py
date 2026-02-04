import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from math import ceil
# plt.style.use('dark_background') # use dark mode style

class LinearAdvection:
    def __init__(self, 
                 phi0 : Callable[[np.array], np.array], 
                 u : float, dt: float, 
                 dx : float, x0 : float, x1 : float):
        
        # Save numerical parameters
        self.x0 = x0
        self.x1 = x1
        self.phi0 = phi0
        self.u = u
        self.dx = dx
        self.dt = dt

        # Define space domain
        self.nx = int((x1 - x0) / dx) + 1
        self.xs = np.linspace(x0, x1, self.nx)

    def plot_phi(self, phi: np.array, time: float):
        label = rf'$t={int(time)}$'
        plt.plot(self.xs, phi, label=label)
    
    def solve_linear(self, t0 : float, t1 : float, display_time : float = 250):
        # Initialize time and streamfunction
        t = t0
        phi_now = self.phi0(self.xs)
        phi_new = phi_now.copy()

        # Plot initial condition
        plt.figure(); plt.clf()
        self.plot_phi(phi_now, 0)

        # Iterate and apply numerical scheme
        while (t < t1):
            # Iterate over each space grid
            for (j, xp) in enumerate(self.xs):
                # Take departure position and apply periodic boundary
                xdep = self.x0 + (xp - self.u * self.dt) % (self.x1 - self.x0)
                # Periodic boundary
                if xdep < self.x0:
                    xdep = self.x1 + xdep
                # Compute nearest grid index to xdep using floor
                m = int(ceil(xdep / self.dx))
                # Compute fraction
                alpha = m - (xdep / self.dx)
                # Right index
                mp = int((m - 1))
                # Apply interpolation
                phi_new[j] = (1 - alpha) * phi_now[m] + alpha * phi_now[mp]

            # Swap values
            phi_now[:] = phi_new[:]

            # Update time
            t += self.dt

            # Plot given condition
            if ((t - t0) % display_time) < self.dt:
                self.plot_phi(phi_now, t)

        # Decorate plot
        self.decorate_plot('linear')

    
    def solve_cubic(self, t0 : float, t1 : float, display_time : float = 250):
        # Initialize time and streamfunction
        t = t0
        phi_now = self.phi0(self.xs)
        phi_new = phi_now.copy()

        # Plot initial condition
        plt.figure(); plt.clf()
        self.plot_phi(phi_now, 0)

        # Iterate and apply numerical scheme
        while (t < t1):
            # Iterate over each space grid
            for (j, xp) in enumerate(self.xs):
                # Take departure position and apply periodic boundary
                xdep = self.x0 + (xp - self.u * self.dt) % (self.x1 - self.x0)
                # Periodic boundary
                if xdep < self.x0:
                    xdep = self.x1 + xdep
                # Compute nearest grid index to xdep using floor
                m = int(ceil(xdep / self.dx))
                # Compute fraction
                alpha = m - (xdep / self.dx)
    
                # Left indexes 
                # By applying periodic boundary, index 0 is the same as nx-1, so index nx is the same as 1
                m_p = int((m + 1) % (self.nx - 1))
                # Right indexe
                # (no need to correct it since Python does periodic calling for negative indexing)
                mp = m - 1
                mpp = m - 2
                

                # Apply interpolation
                phi_new[j] = - alpha * (1 - alpha**2) / 6 * phi_now[mpp]        \
                             + alpha * (1+alpha) * (2-alpha) / 2 * phi_now[mp]  \
                             + (1-alpha**2) * (2-alpha) / 2 * phi_now[m]        \
                             - alpha * (1-alpha) * (2-alpha) / 6 * phi_now[m_p]

            # Swap values
            phi_now = phi_new.copy()

            # Update time
            t += self.dt

            # Plot given condition
            if ((t - t0) % display_time) < self.dt:
                self.plot_phi(phi_now, t)

        # Decorate plot
        self.decorate_plot('cubic')

    def decorate_plot(self, method):
        # Decorate plot
        plt.suptitle(rf'Solution of $\phi$ using SL {method}')
        plt.title(fr'$\Delta x= {self.dx :.2f}, \Delta t={self.dt :.2f}, u={self.u}$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlim(self.x0, self.x1)
        plt.grid(); plt.xlabel('$x$')
        plt.tight_layout()
            
if __name__ == '__main__':
    # Define streamfunction initial condition, phi(x, 0)
    def phi0 (x : np.array) -> np.array:
        # Initialize with zeroes
        out = np.zeros_like(x)
        # First case
        idxs = (x >= 400) & (x < 500)
        out[idxs] = 0.1 * (x[idxs] - 400.0)
        # Second case
        idxs = (x >= 500) & (x < 600)
        out[idxs] = 20.0 - 0.1 * (x[idxs] - 400.0)
        return out
        
    # Define space and time domain, and velocity
    x0, x1  = 0, 1_000
    dx, dt  = 0.5, 1.0
    t0, t1  = 0, 2_000
    u       = 0.75
    dsp     = 250 # display time

    # Solve by linear interpolation
    advc = LinearAdvection(phi0=phi0, u=u, dx=dx, x0=x0, x1=x1, dt=dt)
    advc.solve_linear(t0, t1, display_time=dsp)

    # Solve by cubic interpolation
    advc.solve_cubic(t0, t1, display_time=dsp)

    # Display all cases
    plt.show()