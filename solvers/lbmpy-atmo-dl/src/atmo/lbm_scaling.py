"""lbm_scaling.py

Created 2023-06-15 by C. Lapeyre (lapeyre@cerfacs.fr)

LBMScaling dataclass to help understand LBM inputs
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class LBMScaling:
    ## - General physical parameters - ##
    dx: float = 1  # [m]
    u_max: float = 1  # Expected max velocity [m/s]
    u_bulk_target: float = 1  # Target bulk velocity [m/s]
    ul_bulk_target: float = 0.05  # Target lattice bulk velocity [ø] = [lat/it]
    dt: float = 0.05  # Timestep [s]
    ν_target: float = 1.48e-5  # Best case laminar viscosity [m^2/s]
    ν: float = 1.48e-5  # Actual laminar viscosity [m^2/s]
    νl: float = 1.48e-5  # Lattice viscosity [ø]

    csound: float = 340  # [m/s]

    ## - Numerics. Depends on your configuration, Method...
    # For Cumulants with obstacles: ul_max = 0.05, ω = 1.995 is a good limit
    ul_max: float = 0.05
    ω: float = 1.995
    csoundl: float = 1 / 3**0.5


@dataclass
class StableScaling(LBMScaling):
    """Lattice Boltzmann Scaling helper to ensure stability

    When stability is an issue, it is governed by ul_max and ω.
    The central assumption her is that (ul_max, ω) are fixed by the numerics and
    define a stability limit. The objective is to get close to the limit, but
    not exceed it, by increasing ν from ν_target to a value compatible with
    stability. From a limited set of inputs (dx and u_max essentially) one can
    get the dimensional parameters (dt, actual ν, actual Re) of the simulation.
    """

    @property
    def Mach(self) -> float:
        return self.u_max / self.csound

    def dt_from_dx(self, dx) -> float:
        """Compute dt from dx value"""
        return self.ul_max / self.u_max * dx

    @property
    def dt(self) -> float:
        return self.dt_from_dx(self.dx)

    @dt.setter
    def dt(self, value):
        """Do nothing. Implemented to prevent Attribute Error"""

    @property
    def νl(self):
        return (2 / self.ω - 1) * self.csoundl**2

    @νl.setter
    def νl(self, value):
        """Do nothing. Implemented to prevent Attribute Error"""

    def ν_from_dx(self, dx=None) -> float:
        """Compute ν if dx kept constant"""
        if dx is None:
            dx = self.dx
        return self.νl * dx**2 / self.dt_from_dx(dx)

    @property
    def ν(self) -> float:
        return self.ν_from_dx()

    @ν.setter
    def ν(self, value):
        """Do nothing. Implemented to prevent Attribute Error"""

    def ul_from_u(self, u) -> float:
        """Compute lattice velocity from dimensional one"""
        return u / self.u_max * self.ul_max

    def u_from_ul(self, ul) -> float:
        """Compute physical velocity from lattice one"""
        return ul * self.dx / self.dt

    @property
    def ul_bulk_target(self) -> float:
        return self.ul_from_u(self.u_bulk_target)

    @ul_bulk_target.setter
    def ul_bulk_target(self, value):
        """Do nothing. Implemented to prevent Attribute Error"""

    def t_from_iterations(self, iterations) -> float:
        return self.dt * iterations

    def iterations_from_t(self, t) -> int:
        return int(t / self.dt)

    def stability_line(self, npts=10):
        """Combinations of (ν, dx) that would be stable (and matching dt)"""
        ν_vals = np.geomspace(self.ν_target, self.ν_from_dx(), npts)
        dx_vals = ν_vals / self.νl * self.ul_max / self.u_max
        dt_vals = self.ul_max / self.u_max * dx_vals
        return {"ν": ν_vals, "dx": dx_vals, "dt": dt_vals}

    def Reynolds(self, length, dx=None) -> float:
        if dx is None:
            dx = self.dx
        return length * self.u_bulk_target / self.ν_from_dx(dx)

    def plot_stability_line(self):
        import matplotlib.pyplot as plt

        ν_vals, dx_vals, dt_vals = self.stability_line().values()

        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.loglog(ν_vals, dx_vals, label=f"ul={self.ul_max}, ω={self.ω}")
        ax1.fill_between(ν_vals, 0, dx_vals, alpha=0.5)
        ax2.loglog(ν_vals, dt_vals)

        ax1.set_xlabel("ν (m^2/s)")
        ax1.set_ylabel("dx (m)")
        ax2.set_ylabel("dt (s)")
        ax1.plot(self.ν, dx_vals[-1], "o")
        ax1.annotate(
            "  Requested (ν, dx)",
            xy=(self.ν, dx_vals[0]),
            xytext=(self.ν, dx_vals[-1]),
        )
        ax1.legend()


if __name__ == "__main__":
    sc = StableScaling(dx=0.7, u_max=5, u_bulk_target=2)
    # sc = StableScaling()
    print("An LBM mesh of 0.7 m, a max velocity of 5 m/s")
    print("With ul_max of 0.05 m/s and ω = 1.995")
    print("---")
    print(f"Timestep: {sc.dt:.5f} s")
    print(f"Lowest stable ν: {sc.ν_from_dx():.5f} m^2/s")
    print(f"Max Mach number: {sc.Mach:.5f}")
    print(f"Effective Re number of a 20 m object: {sc.Reynolds(20):.5f}")
