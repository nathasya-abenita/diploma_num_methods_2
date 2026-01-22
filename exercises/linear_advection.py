import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
plt.style.use('dark_background') # use dark mode style

class LinearAdvection:
    def __init__(self, 
                 phi0 : Callable[[np.array], np.array], 
                 u : float, c: float, 
                 dx : float, x0 : float, x1 : float):
        
        # Save numerical parameters
        self.phi0 = phi0
        self.u = u
        self.c = c
        self.dx = dx
        self.dt = c * dx / u # based on definition of Courant number

        # Define space domain
        nx = int((x1 - x0) / dx) + 1
        self.xs = np.linspace(x0, x1, nx)
    
    def ftfs(self, phi_now: np.array) -> np.array:
        return (1 + self.c) * phi_now - self.c * np.roll(phi_now, -1)
    
    def ftbs(self, phi_now: np.array) -> np.array:
        return (1 - self.c) * phi_now + self.c * np.roll(phi_now, 1)

    def plot_phi(self, phi: np.array, time: float):
        label = rf'$t={int(time)}$'
        plt.plot(self.xs, phi, label=label)
    
    def solve(self, t0 : float, t1 : float, display_time : float = 200):
        # Initialize time and streamfunction
        t = t0
        phi = self.phi0(self.xs)

        # Plot initial condition
        plt.figure()
        self.plot_phi(phi, 0)

        # Iterate and apply numerical scheme
        while (t < t1):

            # Update time
            t += self.dt

            # Update streamfunction by using the upwind scheme
            if u > 0:
                phi = self.ftbs(phi)
            else:
                phi = self.ftfs(phi)

            # Plot given condition
            if ((t - t0) % display_time) < self.dt:
                self.plot_phi(phi, t)

        # Decorate plot
        plt.suptitle(r'Solution of $\phi$ using upstream scheme')
        plt.title(fr'$c={self.c}, \Delta x= {self.dx :.2f}, \Delta t={self.dt :.2f}, u={self.u}$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(); plt.xlabel('$x$')
        plt.tight_layout()
            
if __name__ == '__main__':
    # Define streamfunction initial condition, phi(x, 0)
    def phi0 (x : np.array) -> np.array:
        out = np.zeros_like(x)
        idxs = (x >= 40) & (x < 70)
        out[idxs] = np.sin(np.pi * (x[idxs] - 40) / 30) ** 2
        return out
        
    # Define space and time domain, and velocity
    x0, x1  = 0, 100
    dx      = 0.10
    t0, t1  = 0, 1_000
    u       = 0.087

    # Solve for c = 0.1
    advc = LinearAdvection(phi0=phi0, u=u, dx=dx, x0=x0, x1=x1, c=0.1)
    advc.solve(t0, t1)

    # Solve for c = 0.5
    advc = LinearAdvection(phi0=phi0, u=u, dx=dx, x0=x0, x1=x1, c=0.5)
    advc.solve(t0, t1)

    # Solve for c = 1
    advc = LinearAdvection(phi0=phi0, u=u, dx=dx, x0=x0, x1=x1, c=1)
    advc.solve(t0, t1)

    # Display all cases
    plt.show()