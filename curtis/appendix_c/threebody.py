"""Appendix C.2 - ``threebody.m``.

Defines the initial conditions of a planar three-body problem, integrates the
equations of motion with an adaptive Runge-Kutta solver (SciPy's
``solve_ivp`` with method ``RK45`` is the equivalent of MATLAB's ``ode45``)
and reproduces Figures 2.5 and 2.6.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .accel_3body import accel_3body


def threebody(G=6.67259e-20,
              t_initial=0.0,
              t_final=67000.0,
              m=(1.0e29, 1.0e29, 1.0e29),
              r0=((0.0, 0.0), (300000.0, 0.0), (600000.0, 0.0)),
              v0=((0.0, 0.0), (250.0, 250.0), (0.0, 0.0)),
              n_eval=1000):
    """Integrate the planar three-body problem.

    Parameters
    ----------
    G : float
        Gravitational constant (km^3/kg/s^2).
    t_initial, t_final : float
        Initial and final times (s).
    m : sequence
        Masses ``[m1, m2, m3]`` (kg).
    r0 : sequence
        3x2 initial positions; each row is the (x, y) of a mass (km).
    v0 : sequence
        3x2 initial velocities; each row is the (x, y) of a mass (km/s).
    n_eval : int
        Number of output samples between ``t_initial`` and ``t_final``.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        ``t`` (length n) and ``f`` (n x 16). The columns of ``f`` are, in
        order, ``x1, y1, x2, y2, x3, y3, xG, yG, v1x, v1y, v2x, v2y, v3x,
        v3y, vGx, vGy``.
    """
    m = np.asarray(m, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    # Initial position and velocity of the centre of mass.
    rG0 = m @ r0 / m.sum()
    vG0 = m @ v0 / m.sum()

    # Initial conditions as a single 16-component column vector.
    f0 = np.concatenate([r0[0], r0[1], r0[2], rG0,
                         v0[0], v0[1], v0[2], vG0])

    # Pass the initial conditions and time interval to the solver, which
    # uses accel_3body to evaluate the acceleration at each step.  RK45 is
    # the same method as MATLAB's ode45; the tolerances here are tightened
    # below the ode45 defaults (1e-3 / 1e-6) for a more accurate trajectory.
    t_eval = np.linspace(t_initial, t_final, n_eval)
    sol = solve_ivp(accel_3body, [t_initial, t_final], f0,
                    method="RK45", t_eval=t_eval, args=(G, m),
                    rtol=1e-8, atol=1e-8)

    return sol.t, sol.y.T


def plot_threebody(t, f, save_path=None, show=False):
    """Reproduce Figures 2.5 and 2.6 from a threebody solution.

    Parameters
    ----------
    t : numpy.ndarray
        Time samples (unused, kept for symmetry).
    f : numpy.ndarray
        n x 16 solution array from :func:`threebody`.
    save_path : str, optional
        If given, a base path; the two figures are written to
        ``<base>_inertial.png`` and ``<base>_com.png``.
    show : bool
        If True, display the figures interactively.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x1, y1 = f[:, 0], f[:, 1]
    x2, y2 = f[:, 2], f[:, 3]
    x3, y3 = f[:, 4], f[:, 5]
    xG, yG = f[:, 6], f[:, 7]

    # Figure 2.5 - motion relative to the inertial frame.
    fig1, ax1 = plt.subplots()
    ax1.set_title("Figure 2.5: Motion relative to the inertial frame",
                  fontweight="bold", fontsize=12)
    ax1.plot(x1, y1, "r", linewidth=0.5)
    ax1.plot(x2, y2, "g", linewidth=1.0)
    ax1.plot(x3, y3, "b", linewidth=1.5)
    ax1.plot(xG, yG, "--k", linewidth=0.25)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True)
    ax1.axis("equal")

    # Figure 2.6 - motion relative to the centre of mass.
    fig2, ax2 = plt.subplots()
    ax2.set_title("Figure 2.6: Motion relative to the center of mass",
                  fontweight="bold", fontsize=12)
    ax2.plot(x1 - xG, y1 - yG, "r", linewidth=0.5)
    ax2.plot(x2 - xG, y2 - yG, "--g", linewidth=1.0)
    ax2.plot(x3 - xG, y3 - yG, "b", linewidth=1.5)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)
    ax2.axis("equal")

    if save_path:
        fig1.savefig(f"{save_path}_inertial.png", dpi=120,
                     bbox_inches="tight")
        fig2.savefig(f"{save_path}_com.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()

    return fig1, fig2


def _run():
    """Integrate the book's example and save Figures 2.5 and 2.6."""
    import os

    t, f = threebody()
    out = os.path.join(os.path.dirname(__file__), "threebody")
    plot_threebody(t, f, save_path=out)
    print("Integrated planar three-body problem:")
    print(f"  time span      = [{t[0]:g}, {t[-1]:g}] s")
    print(f"  samples        = {len(t)}")
    print(f"  final r1 (km)  = ({f[-1, 0]:.3f}, {f[-1, 1]:.3f})")
    print(f"  final r2 (km)  = ({f[-1, 2]:.3f}, {f[-1, 3]:.3f})")
    print(f"  final r3 (km)  = ({f[-1, 4]:.3f}, {f[-1, 5]:.3f})")
    print(f"  centre of mass = ({f[-1, 6]:.3f}, {f[-1, 7]:.3f})")
    print(f"  figures written to {out}_inertial.png and {out}_com.png")


if __name__ == "__main__":
    _run()
