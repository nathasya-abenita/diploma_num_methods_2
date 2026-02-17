import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
# plt.style.use('dark_background') # use dark mode style

class GravityWave:
    def __init__(self, 
                 u0 : Callable[[np.array], np.array],
                 phi0 : Callable[[np.array], np.array], 
                 mean_phi : float, dt : float,
                 dx : float, x0 : float, x1 : float):
        
        # Save numerical parameters
        self.phi0 = phi0
        self.u0 = u0
        self.dx = dx
        self.dt = dt
        self.dtdx = self.dt / self.dx
        self.c = np.sqrt(mean_phi) * self.dtdx

        # Define space domain
        nx = int((x1 - x0) / dx) + 1
        self.xs = np.linspace(x0, x1, nx)
    
    def ueq_ft(self, u_now: np.array, p_now: np.array) -> np.array:
        return u_now - 0.5 * self.dtdx * (np.roll(p_now, -1) - np.roll(p_now, 1))
    
    def peq_ft(self, u_now: np.array, p_now: np.array) -> np.array:
        return p_now - 0.5 * self.dtdx * (np.roll(u_now, -1) - np.roll(u_now, 1))
    
    def ueq(self, u_old: np.array, p_now: np.array) -> np.array:
        return u_old - self.dtdx * (np.roll(p_now, -1) - np.roll(p_now, 1))
    
    def peq(self, p_old: np.array, u_now: np.array) -> np.array:
        return p_old - self.c * (np.roll(u_now, -1) - np.roll(u_now, 1))

    def plot_phi(self, axs, phi: np.array, u: np.array, time: float):
        label = rf'$t={int(time)}$'
        axs[0].plot(self.xs, phi, label=label)
        axs[1].plot(self.xs, u, label=label)
    
    def solve_three_step_scheme(self, t0: float, t1: float, display_time):
        # Initialize time and stream function
        t = t0
        p_old = self.phi0(self.xs)
        u_old = self.u0(self.xs)

        # Plot initial condition
        fig, axs = plt.subplots(2, sharex=True)
        self.plot_phi(axs, p_old, u_old, 0)

        # Update for the first time before looping
        p_now = self.peq_ft(p_old, u_old)
        u_now = self.ueq_ft(u_old, p_old)
        t += self.dt

        # Iterate and apply numerical scheme
        while (t < t1):

            # Apply scheme
            u_new = self.ueq(u_old, p_now)
            p_new = self.peq(p_old, u_now)

            # Swap
            p_old[:] = p_now
            p_now[:] = p_new
            u_old[:] = u_now
            u_now[:] = u_new

            # Update time
            t += self.dt
            
            # Plot given condition
            if ((t - t0) % display_time) < self.dt:
                self.plot_phi(axs, p_now, u_now, t)

        # Decorate plot
        axs[0].set_title(r'Solution of $\phi$')
        axs[1].set_title(r'Solution of $u$')
        plt.suptitle(rf'Three time steps scheme, $\Delta t={self.dt :.3f}, \Delta x={self.dx :.3f}$')
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[0].grid(); axs[1].grid()
        plt.xlabel('$x$')
        plt.tight_layout()

    def ueq_fo(self, u_now: np.array, p_now: np.array) -> np.array:
        return u_now - 0.5 * self.dtdx * (np.roll(p_now, -1) - np.roll(p_now, 1))
    
    def peq_fo(self, u_fut: np.array, p_now: np.array) -> np.array:
        return p_now - self.c * 0.5 * (np.roll(u_fut, -1) - np.roll(u_fut, 1))
    
    def solve_two_step_scheme(self, t0: float, t1: float, display_time):
        # Initialize time and stream function
        t = t0
        p_now = self.phi0(self.xs)
        u_now = self.u0(self.xs)

        # Plot initial condition
        fig, axs = plt.subplots(2, sharex=True)
        self.plot_phi(axs, p_now, u_now, 0)

        # Iterate and apply numerical scheme
        while (t < t1):

            # Apply scheme (original scheme following the slide)
            u_new = self.ueq_fo(u_now, p_now)
            p_new = self.peq_fo(u_new, p_now)

            # Alternative scheme
            # p_new = self.peq_fo(u_now, p_now)
            # u_new = self.ueq_fo(u_now, p_new)

            # Swap
            p_now[:] = p_new
            u_now[:] = u_new

            # Update time
            t += self.dt
            
            # Plot given condition
            if ((t - t0) % display_time) < self.dt:
                self.plot_phi(axs, p_now, u_now, t)

        # Decorate plot
        axs[0].set_title(r'Solution of $\phi$')
        axs[1].set_title(r'Solution of $u$')
        plt.suptitle(rf'Two time steps scheme, $\Delta t={self.dt :.3f}, \Delta x={self.dx :.3f}$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[0].grid(); axs[1].grid()
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
        out = fill_array(out, x, [400.0, 600.0], lambda x : np.sin((((x - 400) / 200) * np.pi)) ** 2)
        return out
    
    def u0 (x: np.array) -> np.array:
        out = np.zeros_like(x)
        return out
        
    # Define space and time domain, and velocity
    x0, x1      = 0, 1_000
    dx          = 0.5
    dt          = 0.1
    t0, t1      = 0, 2_000
    mean_phi    = 1             # mean height

    # Create class instance
    advc = GravityWave(u0=u0, phi0=phi0, mean_phi=mean_phi, dt=dt, dx=dx, x0=x0, x1=x1)
    
    # Solve (three time steps scheme)
    advc.solve_three_step_scheme(t0, t1, display_time=200)

    # Solve (two time steps scheme)
    advc.solve_two_step_scheme(t0, t1, display_time=200)

    # Display solution
    plt.show()
    