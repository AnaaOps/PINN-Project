"""
Reference (ground-truth) solution of Burgers' equation via the Cole-Hopf transform.

    u_t + u u_x = nu u_xx,   x in [-1,1], t in [0,1]
    u(x,0) = -sin(pi x),     u(+-1,t) = 0,   nu = 0.01/pi

Closed form (Basdevant et al. 1986):

    u(x,t) = - INT sin(pi(x-eta)) f(x-eta) exp(-eta^2/(4 nu t)) d eta
             ---------------------------------------------------------
             INT            f(x-eta) exp(-eta^2/(4 nu t)) d eta

    f(y) = exp(-cos(pi y) / (2 pi nu))

Evaluated with dense trapezoid quadrature in log-space (numerically stable,
no solver / no external data file needed). numpy only.
"""
import numpy as np

NU = 0.01 / np.pi


def burgers_exact(x, t, nu=NU, n_eta=20001, eta_max=1.5):
    """x: 1D array, t: scalar > 0  ->  u(x, t) as 1D array."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if t <= 0:
        return -np.sin(np.pi * x)
    eta = np.linspace(-eta_max, eta_max, n_eta)
    d_eta = eta[1] - eta[0]
    y = x[:, None] - eta[None, :]
    log_w = -np.cos(np.pi * y) / (2 * np.pi * nu) - eta[None, :] ** 2 / (4 * nu * t)
    log_w -= log_w.max(axis=1, keepdims=True)      # stabilise
    w = np.exp(log_w)
    num = (-np.sin(np.pi * y) * w).sum(axis=1) * d_eta
    den = w.sum(axis=1) * d_eta
    return num / den


def burgers_grid(nx=256, nt=101, nu=NU):
    """Full reference field. Returns x (nx,), t (nt,), U (nx, nt)."""
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    U = np.stack([burgers_exact(x, ti, nu) for ti in t], axis=1)
    return x, t, U


if __name__ == "__main__":
    # self-test: IC, BC, and PDE residual by finite differences
    x, t, U = burgers_grid(nx=801, nt=201)
    print("IC error  :", np.abs(U[:, 0] + np.sin(np.pi * x)).max())
    print("BC values :", abs(U[0, 1:]).max(), abs(U[-1, 1:]).max())
    dx, dt = x[1] - x[0], t[1] - t[0]
    ut = (U[1:-1, 2:] - U[1:-1, :-2]) / (2 * dt)
    ux = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * dx)
    uxx = (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / dx**2
    res = ut + U[1:-1, 1:-1] * ux - NU * uxx
    # skip the first few time steps and the shock cell for a clean number
    print("median |PDE residual|:", np.median(np.abs(res[:, 5:])))
    i0 = np.argmin(np.abs(x)); 
    print("u(x=0.0, t=0.5) =", U[i0, 100], " min slope at t=1:", ux[:, -1].min())
