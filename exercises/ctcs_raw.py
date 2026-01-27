import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
# plt.style.use('dark_background') # use dark mode style

class CTCS_RAW:
    def __init__(self, 
                 phi0 : Callable[[np.array], np.array], 
                 u : float, c: float, 
                 dx : float, x0 : float, x1 : float,
                 alpha : float = 0.0,
                 beta : float = 1):
        
        # Save numerical parameters
        self.phi0 = phi0
        self.u = u
        self.c = c
        self.dx = dx
        self.dt = c * dx / u # based on definition of Courant number

        # Save filter info
        self.alpha = alpha
        self.beta = beta

        # Define space domain
        nx = int((x1 - x0) / dx) + 1
        self.xs = np.linspace(x0, x1, nx)
    
    def ftfs(self, phi_now: np.array) -> np.array:
        return (1 + self.c) * phi_now - self.c * np.roll(phi_now, -1)
    
    def ftbs(self, phi_now: np.array) -> np.array:
        return (1 - self.c) * phi_now + self.c * np.roll(phi_now, 1)

    def ctcs(self, phi_old: np.array, phi_now: np.array) -> np.array:
        return phi_old - self.c * (np.roll(phi_now, -1) - np.roll(phi_now, 1))

    def plot_phi(self, phi: np.array, time: float):
        label = rf'$t={int(time)}$'
        plt.plot(self.xs, phi, label=label)
    
    def solve(self, t0: float, t1: float, display_time: float = 200):
        # Initialize time and streamfunction
        t = t0
        phi_old = self.phi0(self.xs)

        # Plot initial condition
        plt.figure(figsize=(12, 6))
        self.plot_phi(phi_old, 0)

        # Update streamfunction by using the upwind scheme
        # for the first time step
        if u > 0:
            phi_now = self.ftbs(phi_old)
        else:
            phi_now = self.ftfs(phi_old)
        t += self.dt

        # Iterate and apply numerical scheme
        while (t < t1):

            # Update time
            t += self.dt
           
            # Update streamfunction by using the upwind scheme
            phi_new = self.ctcs(phi_old, phi_now)

            # Swap values and apply filter
            d = self.alpha * (phi_old + phi_new - 2.0 * phi_now)
            phi_old = phi_now + self.beta * d
            phi_now = phi_new + (1 - self.beta) * d
            
            # Plot given condition
            if ((t - t0) % display_time) < self.dt:
                self.plot_phi(phi_new, t) # Question: Should we plot phi_new or phi_now?

        # Decorate plot
        plt.suptitle(r'Solution of $\phi$ using leapfrog scheme')
        plt.title(rf'$c={self.c}, \alpha = {self.alpha :.2f}, \beta = {self.beta: .2f}, \Delta t={self.dt :.2f}, u={self.u}$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('$x$')
        plt.tight_layout()

def fill_array(out: np.array, x: np.array, bounds: list[float, float], val: float) -> np.array:
    idxs = (x >= bounds[0]) & (x < bounds[1])
    out[idxs] = val
    return out
            
if __name__ == '__main__':
    # Define streamfunction initial condition, phi(x, 0)
    def phi0 (x: np.array) -> np.array:
        out = np.full(x.shape, 0.1)
        out = fill_array(out, x, [200.0, 250.0], 2.0)
        out = fill_array(out, x, [250.0, 300.0], 1.0)
        return out
        
    # Define space and time domain, and velocity
    x0, x1  = 0, 500
    dx      = 0.1
    t0, t1  = 0, 1_000
    u       = -0.31

    # Define Courant number of choice
    c       = -0.4

    # Solve without filter
    advc = CTCS_RAW(phi0=phi0, u=u, dx=dx, x0=x0, 
                    x1=x1, c=c)
    advc.solve(t0, t1)

    # Solve with RA filter
    advc = CTCS_RAW(phi0=phi0, u=u, dx=dx, x0=x0, 
                    x1=x1, c=c, alpha=0.1)
    advc.solve(t0, t1)

    # Solve with RAW filter
    advc = CTCS_RAW(phi0=phi0, u=u, dx=dx, x0=x0, 
                    x1=x1, c=c, alpha=0.05, beta=0.53)
    advc.solve(t0, t1)

    # Display solution
    plt.show()
    