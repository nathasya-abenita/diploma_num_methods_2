import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
plt.style.use('dark_background') # use dark mode style

class Diffusion:
    def __init__(self, 
                 phi0 : Callable[[np.array], np.array], 
                 k : float, dt: float, temp0 : float,
                 dx : float, x0 : float, x1 : float):
        # Save numerical parameters
        self.x0 = x0
        self.x1 = x1
        self.phi0 = phi0
        self.k = k
        self.dx = dx
        self.dt = dt
        self.temp0 = temp0

        # Define space domain
        self.nx = int((x1 - x0) / dx) + 1
        self.xs = np.linspace(x0, x1, self.nx)

    def plot_phi(self, phi: np.array, time: float):
        label = rf'$t={int(time)}$'
        plt.plot(self.xs, phi, label=label)

    def diff_tdma (self, phi_now : np.array,
                   A : np.array, B: np.array, C : np.array, F : np.array, 
                   delta : np.array) -> np.array:
        
        # Initialize new streamfunction
        phi_new = np.empty_like(phi_now)

        # Initialize values for F and delta
        F[0] = 0
        delta[0] = self.temp0

        # Apply forward elimination
        for j in range (len(C)):
            F[j + 1] = C[j] / (B[j + 1] - A[j] * F[j])
            delta[j + 1] = (phi_now[j + 1] - A[j] * delta[j]) / \
                            (B[j + 1] - A[j] * F[j])
                            
        # Apply boundary condition
        phi_new[-1] = self.temp0

        # Back substitution
        for j in range (len(F) - 1, -1, -1):
            phi_new[j] = delta[j] - F[j] * phi_new[j+1]
        return phi_new
    
    def solve_tdma(self, t0 : float, t1 : float, display_time):
        # Define multiplification factor
        alpha = self.k * self.dt / self.dx ** 2

        # Initialize time and streamfunction
        t = t0
        phi_now = self.phi0(self.xs)

        # Plot initial condition
        plt.figure(); plt.clf()
        self.plot_phi(phi_now, 0)

        # Create constant arrays and initialize empty array for F and delta
        A = np.full(self.nx - 2, -alpha)
        C = np.full(self.nx - 2, -alpha)
        B = np.full(self.nx - 1, 1 + 2 * alpha)
        F = np.empty(self.nx - 1)
        delta = np.empty(self.nx - 1)

        # Iterate and apply numerical scheme
        while (t < t1):
            # Call diffusion
            phi_new = self.diff_tdma(phi_now, A, B, C, F, delta)

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
    x0, x1  = 0, 1              # [m]
    dx      = 0.01              # [m]
    t0, t1  = 0, 6 * 60 * 60    # [s]
    k       = 2.9e-5            # [m^2/s]
    dt      = 1800              # [s]
    temp0   = 273.15            # imposed temperature in the boundary [K]

    # Solve by linear interpolation
    advc = Diffusion(phi0=phi0, k=k, dx=dx, x0=x0, x1=x1, dt=dt, temp0=temp0)
    advc.solve_tdma(t0, t1, display_time=1 * 60 * 60)

    # Display results
    plt.show()