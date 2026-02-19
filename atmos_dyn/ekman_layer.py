import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from math import ceil
# plt.style.use('dark_background') # use dark mode style

class Diffusion:
    def __init__(self, u0_func, v0_func, ug, vg, z0, z1, dz, km, c, lat):
        # Save numerical parameters
        self.z0 = z0
        self.z1 = z1
        self.ug = ug
        self.vg = vg
        self.u0_func = u0_func
        self.v0_func = v0_func
        self.km = km
        self.c = c
        self.dt = c * dz ** 2 / km
        self.dz = dz

        # Compute planetary vorticity
        omega = 7.29e-5 # s^{-1}
        self.f = 2 * omega * np.sin(lat * np.pi / 180)

        # Define space domain
        self.nz = int((z1 - z0) / dz) + 1
        self.zs = np.linspace(z0, z1, self.nz)

        # Other parameters
        self.day_secs = 3600 * 24
        self.gamma = np.sqrt(self.f / (2 * km)) # for analytical solution

    def update_us(self, us, vs):
        us_new = np.empty_like(us)
        
        # Boundary conditions (Dirchlet)
        us_new[0] = 0
        us_new[-1] = self.ug

        # Apply scheme
        us_new[1:-1] = us[1:-1] + c * (us[2:] - 2 * us[1:-1] + us[:-2]) + \
                       self.f * (vs[1:-1] - self.vg)
                
        return us_new
    
    def update_vs(self, us, vs):
        vs_new = np.empty_like(vs)
        
        # Boundary conditions (Dirchlet)
        vs_new[0] = 0
        vs_new[-1] = self.vg

        # Apply scheme
        vs_new[1:-1] = vs[1:-1] + c * (vs[2:] - 2 * vs[1:-1] + vs[:-2]) - \
                       self.f * (us[1:-1] - self.ug)
                
        return vs_new

    def plot(self, us: np.array, vs: np.array, time: float):
        label = rf'$t={int(time / self.day_secs)}$ days'
        plt.plot(us, vs, label=label)

    def solve(self, t0 : float, t1 : float, display_time):

        # Initialize time and streamfunction
        t = t0
        us_now = self.u0_func(self.zs)
        vs_now = self.v0_func(self.zs)

        # Initialize figure
        plt.figure()

        # Iterate and apply numerical scheme
        while (t < t1):
            # Apply scheme
            us_new = self.update_us(us_now, vs_now)
            vs_new = self.update_vs(us_now, vs_now)

            # Swap values
            us_now[:] = us_new
            vs_now[:] = vs_new

            # Update time
            t += self.dt

            # Plot given condition
            if ((t - t0) % display_time) < self.dt:
                self.plot(us_now, vs_now, t)
        
        # Decorate plot
        self.decorate_plot()
        return us_now, vs_now
    
    def decorate_plot(self):
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(r'$u \:[ms^{-1}]$'); plt.ylabel(r'$v \:[ms^{-1}]$')
        plt.suptitle('Numerical simulation')
        plt.title(rf"$\Delta t={self.dt :.3f} \:s, \Delta z={self.dz :.3f} \:m$")
        plt.tight_layout()

    def compute_us_analytical (self):
        return self.ug - np.exp(-self.gamma * self.zs) * \
            (self.ug * np.cos(self.gamma * self.zs) + self.vg * np.sin(self.gamma * self.zs))

    def compute_vs_analytical (self):
        return self.vg + np.exp(-self.gamma * self.zs) * \
            (self.ug * np.sin(self.gamma * self.zs) - self.vg * np.cos(self.gamma * self.zs))

    def plot_compare(self, us_num, vs_num):
        plt.figure()
        
        # Analytical solution
        us_anl = self.compute_us_analytical()
        vs_anl = self.compute_vs_analytical()
        plt.plot(us_anl, vs_anl, label='Analytical', color='k')
        # Numerical solution
        plt.plot(us_num, vs_num, '--', label='Numerical', color='tab:red')
        # Compute RMSE
        rmse_u = np.sqrt(np.sum((us_anl - us_num) ** 2) / self.nz)
        rmse_v = np.sqrt(np.sum((vs_anl - vs_num) ** 2) / self.nz)
        # Decorate
        plt.legend()
        plt.grid()
        plt.xlabel(r'$u \:[ms^{-1}]$'); plt.ylabel(r'$v \:[ms^{-1}]$')
        plt.suptitle('Comparison between numerical and analytical solution')
        plt.title(rf'$RMSE(u)={rmse_u :.3f} \:m/s, RMSE(v)={rmse_v :.3f} \:m/s$')
        plt.tight_layout()

if __name__ == '__main__':

    day_secs = (3600) * 24
    
    # Define geostrophic flow
    ug, vg = 10, 0 # m/s

    # Define initial condition 
    u0_func = lambda zs : np.full_like(zs, ug) # m/s
    v0_func = lambda zs : np.full_like(zs, 0)
        
    # Define space and time domain, and diffusion coefficient
    z0, z1  = 0, 3_000          # m
    dz      = 100               # m
    t0, t1  = 0, day_secs * 18   # s
    km      = 5                 # m^2 s^{-1}
    c       = 0.001                # Courant number
    lat     = 45                # deg

    # Solve by linear interpolation
    diff = Diffusion(u0_func, v0_func, ug, vg, z0, z1, dz, km, c, lat)
    us, vs = diff.solve(t0, t1, display_time=day_secs* 2)
    diff.plot_compare(us, vs)

    # Display results
    plt.show()