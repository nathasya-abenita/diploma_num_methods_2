import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

class PotentialVorticity:
    def __init__ (self, re : float, phi : float, lx : float, ly : float, 
                  d : float, dt : float, H : float, 
                  psi0 : Callable[[float, float], float], ht : Callable[[float, float], float]):
        # Save parameters
        self.re = re
        self.phi = phi
        self.lx, self.ly = lx, ly
        self.d = d
        self.dt = dt
        self.H = H
        self.psi0 = psi0
        
        # Create spatial domain
        self.nxy = int(lx / d) + 1
        x = np.linspace(0, lx, self.nxy)
        y = np.linspace(0, ly, self.nxy)
        self.xv, self.yv = np.meshgrid(x, y, indexing='ij')

        # Common parameters
        self.beta = 1e-11 * 60 * 60 # (m/hr)
        omega = 7.29 * 1e-5 * 60 * 60 # (hr^{-1})
        self.f0 = 2 * omega * np.sin(phi * np.pi / 180)

        # Compute total height of fluid
        self.h = H - ht(self.xv, self.yv)
        self.ht = ht(self.xv, self.yv)
    
    def compute_xi(self, psi):
        # Initialize xi with zeroes
        xi = np.zeros_like(psi)

        # Correct meridional boundary condition (y = 0)
        # xi[:, 0] = self.psi0(self.xv, 0)

        xi[:, 1:-1] = (
            np.roll(psi[:, 1:-1],  1, axis=0) + 
            np.roll(psi[:, 1:-1], -1, axis=0) + 
            psi[:, 2:] + 
            psi[:, :-2] - 
            4 * psi[:, 1:-1]
            ) / self.d**2

        return xi
        
    def compute_u (self, psi):
        u = np.zeros_like(psi)
        u[:, 1:-1] = - (psi[:, 2:] - psi[:, :-2]) / (2*self.d)
        return u
    
    def compute_v(self, psi):
        v = (np.roll(psi, 1, axis=0) - np.roll(psi, -1, axis=0)) / (2 * self.d)
        # v[:, 0] = 0
        # v[:, -1] = 0
        return v
    
    def compute_F (self, xi, u, v): 
        F = np.empty_like(xi)
        F[:, 1:-1] = -((np.roll(u[:, 1:-1], 1, axis=0) * np.roll(xi[:, 1:-1], 1, axis=0) - np.roll(u[:, 1:-1], -1, axis=0) * np.roll(xi[:, 1:-1], -1, axis=0)) + (v[:, 2:] * xi[:, 2:] - v[:, :-2] * xi[:, :-2])) / (2 * self.d) - \
            self.beta * v[:, 1:-1] + \
            self.f0 / (2 * self.d * np.roll(self.h[:, 1:-1], 1, axis=0)) * ( (np.roll(u[:, 1:-1], 1, axis=0) * np.roll(self.h[:, 1:-1], 1, axis=0) - np.roll(u[:, 1:-1], -1, axis=0) * np.roll(self.h[:, 1:-1], -1, axis=0)) + (v[:, 2:] * self.h[:, 2:] - v[:, :-2] * self.h[:, :-2]) )
        return F

    def solve (self, T : float):
        # Initial condition of streamfunction
        psi_now = self.psi0(self.xv, self.yv)
        # self.psi_0 = psi_now[:, 0]
        self.display(psi_now, 0)

        # Initial time
        t = 0 

        # Iterate until end of simulation
        while (t < T):
            print(f'evaluating at lapsed time of {t+self.dt} hr')
            # Evaluate current relative vorticity
            xi_now = self.compute_xi(psi_now)

            # Determine velocities
            u_now = self.compute_u(psi_now)
            v_now = self.compute_v(psi_now)

            # Evaluate F_{i,j}
            F_now = self.compute_F(xi_now, u_now, v_now)

            # Update relative vorticity
            xi_now[:, 1:-1] = (xi_now[:, 1:-1] + self.dt * F_now[:, 1:-1]) / (1 + self.dt * self.re)

            # Update streamfunction
            psi_now = self.invert_xi_to_psi(xi_now)

            # Update time
            t += self.dt

            self.display(psi_now, t)
        return psi_now
    
    def display (self, psi, T, title=r'$\psi$'):
        # plt.figure(); 
        plt.clf()
        # Add contour lines
        contour_lines = plt.contour(self.xv, self.yv, self.ht, levels=5, colors='k')
        plt.clabel(contour_lines, inline=True, fontsize=10, fmt="%.1f")
        
        # Color matrix
        mesh = plt.pcolormesh(self.xv, self.yv, psi, shading='auto', cmap='viridis', vmin=0, vmax=1e4)
        
        # Activate colorbars
        plt.colorbar(mesh)

        # Texts
        plt.suptitle(title)
        plt.title(rf'$T={T}$ hr')
        plt.xlabel(r'$x$'); plt.ylabel(r'$y$')

        plt.pause(0.1)

    def invert_xi_to_psi(self, xi, niter=1_000, tol=1e-6, omega=1.1):
        psi = np.zeros_like(xi, dtype=np.float64)

        for _ in range(niter):
            # Update interior points only
            for j in range(1, self.nxy-1):        # meridional index
                for i in range(self.nxy):         # zonal index (periodic)
                    ip = (i + 1) % self.nxy
                    im = (i - 1) % self.nxy

                    psi_new = 0.25 * (
                        psi[j, ip] + psi[j, im] +
                        psi[j+1, i] + psi[j-1, i] -
                        self.d ** 2 * xi[j, i]
                    )

                    delta = psi_new - psi[j, i]
                    psi[j, i] += omega * delta

                    max_update = abs(delta)

            if max_update < tol:
                break

        return psi

if __name__ == '__main__':
    # Define initial condition for streamfunction
    def psi0_general (x : np.array, y : np.array, ly : float) -> np.array :
        return -10 * (y - ly)
    
    # Define mountain shape function
    def ht_general (x : np.array, y : np.array, 
                    lx : float, ly : float,
                    h0 : float, n_mt : int) -> np.array:
        return h0 * np.sin(n_mt * 2 * np.pi * x / lx) * np.sin(np.pi * y / ly)

    # Define numerical parameters
    re  = 1 / 24    # Ekman pumping term (hr^{-1})
    phi = 45        # latitude (deg N)
    lx  = 2e7       # zonal width (m)
    ly  = 3e6       # meridional width (m)
    d   = 1e6/2       # zonal and meridional step size (m)
    dt  = 1         # time step size (hr)
    H   = 1.2e4     # fixed TOA height (m)
    T   = 24        # simulation duration (hr)

    # Case (a)
    n_mt = 2
    psi0 = lambda x, y : psi0_general(x, y, ly)
    ht = lambda x, y : ht_general(x, y, lx, ly, h0=1e3, n_mt=n_mt) 
    pv = PotentialVorticity(re, phi, lx, ly, d, dt, H, psi0, ht)
    psi_end = pv.solve(T)
    plt.show()
    # plt.savefig(rf'vorticity_N_{n_mt}.gif')