import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
# plt.style.use('dark_background') # use dark mode style

class AdvectionDiffusion:
    def __init__(self, 
                 phi0 : Callable[[np.array], np.array], 
                 u : float, c: float, k : float,
                 dx : float, x0 : float, x1 : float,
                 alpha : float = 0.0,
                 beta : float = 1):
        
        # Save numerical parameters
        self.phi0 = phi0
        self.u = u
        self.k = k
        self.dx = dx
        self.dt = min([c * dx / u, abs(c) * dx ** 2 / k]) # based on definition of Courant number
        self.dc = k * self.dt / dx ** 2
        self.c = u * self.dt / dx

        # Save filter info
        self.alpha = alpha
        self.beta = beta

        # Define space domain
        nx = int((x1 - x0) / dx) + 1
        self.xs = np.linspace(x0, x1, nx)
    
    def ftcs_diff(self, phi_now: np.array) -> np.array:
        return phi_now - 0.5 * self.c * (np.roll(phi_now, -1) - np.roll(phi_now, 1)) + \
        self.dc * (np.roll(phi_now, -1) - 2 * phi_now + np.roll(phi_now, 1))
    
    def ctcs_diff(self, phi_old: np.array, phi_now: np.array) -> np.array:
        return phi_old - self.c * (np.roll(phi_now, -1) - np.roll(phi_now, 1)) + \
        2 * self.dc * (np.roll(phi_old, -1) - 2 * phi_old + np.roll(phi_old, 1))

    def plot_phi(self, phi: np.array, time: float):
        label = rf'$t={int(time)}$'
        plt.plot(self.xs, phi, label=label)
    
    def solve(self, t0: float, t1: float, display_time):
        # Initialize time and stream function
        t = t0
        phi_old = self.phi0(self.xs)

        # Plot initial condition
        plt.figure()
        self.plot_phi(phi_old, 0)

        # Update stream function by FTCS
        phi_now = self.ftcs_diff(phi_old)
        t += self.dt

        # Iterate and apply numerical scheme
        while (t < t1):

            # Update stream function by using CTCS
            phi_new = self.ctcs_diff(phi_old, phi_now)

            # Swap values and apply filter
            d = self.alpha * (phi_old + phi_new - 2.0 * phi_now)
            phi_old = phi_now + self.beta * d
            phi_now = phi_new + (1 - self.beta) * d

            # Update time
            t += self.dt
            
            # Plot given condition
            if ((t - t0) % display_time) < self.dt:
                self.plot_phi(phi_new, t)

        # Decorate plot
        plt.suptitle(r'Solution of $\phi$')
        plt.title(rf'$\Delta t={self.dt :.3f}, \Delta x={self.dx :.3f}, u={self.u}, K={self.k :.2f}$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.xlabel('$x$')
        plt.tight_layout()

def fill_array(out: np.array, x: np.array, bounds: list[float, float], f) -> np.array:
    idxs = (x >= bounds[0]) & (x < bounds[1])
    out[idxs] = f(x[idxs])
    return out
            
if __name__ == '__main__':
    # Define streamfunction initial condition, phi(x, 0)
    def phi0 (x: np.array) -> np.array:
        out = np.zeros_like(x)
        out = fill_array(out, x, [400.0, 500.0], lambda x : 0.01 * (x-400))
        out = fill_array(out, x, [500.0, 600.0], lambda x : 2 - 0.01 * (x-400))
        return out
        
    # Define space and time domain, and velocity
    x0, x1  = 0, 1_000
    t0, t1  = 0, 2_000
    u       = 0.95
    k       = 0.029

    # Define Courant number of choice
    c       = 0.35

    # Solve with RAW filter (dx = 0.2)
    advc = AdvectionDiffusion(phi0=phi0, u=u, dx=0.2, x0=x0, 
                    x1=x1, c=c, k=k, alpha=0.1, beta=0.53)
    advc.solve(t0, t1, display_time=500)

    # Solve with RAW filter (dx = 0.05)
    advc = AdvectionDiffusion(phi0=phi0, u=u, dx=0.05, x0=x0, 
                    x1=x1, c=c, k=k, alpha=0.1, beta=0.53)
    advc.solve(t0, t1, display_time=500)

    # Display solution
    plt.show()
    