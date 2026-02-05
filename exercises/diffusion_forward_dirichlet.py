import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from math import ceil
plt.style.use('dark_background') # use dark mode style

class Diffusion:
    def __init__(self, 
                 phi0 : Callable[[np.array], np.array], 
                 k : float, c: float, temp0 : float,
                 dx : float, x0 : float, x1 : float):
        # Save numerical parameters
        self.x0 = x0
        self.x1 = x1
        self.phi0 = phi0
        self.k = k
        self.dx = dx
        self.dt = (c * dx**2) / (2*k)
        self.temp0 = temp0

        # Define space domain
        self.nx = int((x1 - x0) / dx) + 1
        self.xs = np.linspace(x0, x1, self.nx)

    def plot_phi(self, phi: np.array, time: float):
        label = rf'$t={int(time)}$'
        plt.plot(self.xs, phi, label=label)

    def diff (self, mult : float, phi_now : np.array) -> np.array:
        phi_new = np.zeros_like(phi_now)
        # Apply boundary condition
        phi_new[0], phi_new[-1] = self.temp0, self.temp0
        # Apply scheme
        phi_new[1:-1] = phi_now[1:-1] + \
            mult * (phi_now[2:] - 2 * phi_now[1:-1] + phi_now[:-2])
        return phi_new
    
    def solve(self, t0 : float, t1 : float, display_time):
        # Define multiplification
        mult = self.k * self.dt / self.dx ** 2

        # Initialize time and streamfunction
        t = t0
        phi_now = self.phi0(self.xs)

        # Plot initial condition
        plt.figure(); plt.clf()
        self.plot_phi(phi_now, 0)

        # Iterate and apply numerical scheme
        while (t < t1):
            # Call diffusion
            phi_new = self.diff(mult, phi_now)

            # Swap values
            phi_now[:] = phi_new[:]

            # Update time
            t += self.dt

            # Plot given condition
            if ((t - t0) % display_time) < self.dt:
                self.plot_phi(phi_now, t)

        # Decorate plot
        self.decorate_plot()

    def decorate_plot(self):
        # Decorate plot
        plt.suptitle(rf'Solution of $\phi(x,t)$')
        plt.title(fr'$\Delta x= {self.dx :.2f}, \Delta t={self.dt :.2f}, k={self.k}$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlim(self.x0, self.x1)
        plt.grid(); plt.xlabel(r'$x$'); plt.ylabel(r'$\phi$')
        plt.tight_layout()
            
if __name__ == '__main__':
    # Define streamfunction initial condition, phi(x, 0)
    def phi0 (x : np.array) -> np.array:
        # Initialize with zeroes
        out = np.zeros_like(x)
        # First case
        idxs = (x >= 0) & (x <= 0.5)
        out[idxs] = 273.15 + 20 * x[idxs] + np.sin(50 * np.pi * x[idxs])
        # Second case
        idxs = (x > 0.5) & (x <= 1)
        out[idxs] = 273.15 + 20 - 20 * x[idxs] + np.sin(50 * np.pi * x[idxs])
        return out
        
    # Define space and time domain, and diffusion coefficient
    x0, x1  = 0, 1      # [m]
    dx      = 0.01
    t0, t1  = 0, 6 * 60 * 60
    k       = 2.9e-5    # [m^2/s]
    c       = 0.9       # Courant number
    temp0   = 273.15    # imposed temperature in the boundary

    # Solve by linear interpolation
    advc = Diffusion(phi0=phi0, k=k, dx=dx, x0=x0, x1=x1, c=c, temp0=temp0)
    advc.solve(t0, t1, display_time=1 * 60 * 60)

    # Display results
    plt.show()