import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

class Integrate:
    def __init__ (self, 
                  dfunc: Callable[[float, float], float], 
                  solfunc: Callable[[float], float]):
        self.dfunc = dfunc
        self.solfunc = solfunc

    def integrate_euler(self, x: float, y: float, dx: float) -> float:
        return y + dx * self.dfunc(x, y)

    def integrate_heun(self, x: float, y: float, dx: float) -> float:
        f0 = self.dfunc(x, y)
        ystar = y + dx * f0
        xp = x + dx
        return y + 0.5 * dx * (f0 + self.dfunc(xp, ystar))
    
    def perform_euler(self, x: float, y: float, dx: float, nx: int) -> float:
        # Initialize solution array and initial condition
        sols = np.empty(nx)
        sols[0] = y
        errors = np.zeros(nx)

        # Perform integration
        for i in range (1, nx):
            # Call integration
            y = self.integrate_euler(x, y, dx)

            # Update x
            x = x + dx

            # Save solution
            sols[i] = y
            errors[i] = y - self.solfunc(x)
        return sols, errors
    
    def perform_heun (self, x: float, y: float, dx: float, nx: int) -> float:
        # Initialize solution array
        sols = np.empty(nx)
        sols[0] = y
        errors = np.empty(nx)

        # Perform integration
        for i in range (1, nx):
            # Call integration
            y = self.integrate_heun(x, y, dx)

            # Update x
            x = x + dx

            # Save solution
            sols[i] = y
            errors[i] = y - self.solfunc(x)
        return sols, errors

if __name__ == '__main__':
    # Define problem by the known derivative where y is in x
    dfunc = lambda x, y : -0.5 * y + 4 * np.exp(-0.5 * x) * np.cos(4.0 * x)

    # Define analytical solution
    solfunc = lambda x : np.exp(-0.5 * x) * np.sin(4.0 * x)

    # Define initial conditions
    x0, x1 = 0.0, 10.0  # Interval of x
    dx = 0.1            # Time step size
    y0 = 0.0            # Initial condition of y at t = 0

    # Compute space intervals
    nx = int( (x1 - x0) / dx ) + 1

    # Call class
    intg = Integrate(dfunc, solfunc)

    # Perform integration
    sols_euler, sols_e_euler = intg.perform_euler(x0, y0, dx, nx)
    sols_heun, sols_e_heun = intg.perform_heun(x0, y0, dx, nx)

    # Plot solution
    xs = np.linspace(x0, x1, nx)
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axs[0].plot(xs, sols_euler, label='Euler scheme')
    axs[0].plot(xs, sols_heun, label='Heun scheme')
    axs[0].plot(xs, solfunc(xs), '--k', label='Analytical solution')
    axs[0].set_title(rf'Solution with $dx={dx}$')
    axs[0].set_xlabel('$x$'); axs[0].set_ylabel('$y$')
    axs[0].legend(); axs[0].grid()

    # Plot error
    axs[1].plot(xs, sols_e_euler, label='Euler scheme')
    axs[1].plot(xs, sols_e_heun, label='Heun scheme')
    axs[1].set_title(rf'Error with $dx={dx}$')
    axs[1].set_xlabel('$x$'); axs[1].set_ylabel('$e_j$')
    axs[1].legend(); axs[1].grid()
    plt.show()