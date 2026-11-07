"""
FNO dataset: periodic 1D viscous Burgers  u_t + u u_x = nu u_xx,  nu = 0.01
x in [0,1) periodic, t in [0,1].  Operator to learn: u(x,0) -> u(x,1)

Random smooth initial conditions; ground truth from a pseudo-spectral solver
(FFT in space, integrating-factor RK4 in time, 2/3 dealiasing).

usage:  python FNO_dataset.py          # writes data/burgers_fno.npz
        python FNO_dataset.py --check  # solver sanity checks only
"""
import argparse
import os
import numpy as np

NU = 0.01
NX = 128


def random_initial_conditions(n_samples, nx=NX, alpha=2.5, tau=5.0, seed=0):
    """smooth random periodic functions; spectrum ~ (k^2 + tau^2)^(-alpha/2)"""
    rng = np.random.default_rng(seed)
    k = np.arange(nx // 2 + 1)
    amp = (k**2 + tau**2) ** (-alpha / 2)
    amp[0] = 0.0                                   # zero-mean
    coef = (rng.standard_normal((n_samples, k.size))
            + 1j * rng.standard_normal((n_samples, k.size))) * amp
    u0 = np.fft.irfft(coef, n=nx, axis=1)
    u0 /= u0.std(axis=1, keepdims=True)            # unit std per sample
    return u0


def solve_burgers(u0, nu=NU, t_final=1.0, dt=5e-4):
    """pseudo-spectral solver, integrating-factor RK4. u0: (B, nx) -> (B, nx)"""
    nx = u0.shape[1]
    k = 2 * np.pi * np.fft.rfftfreq(nx, d=1.0 / nx)      # wavenumbers on [0,1)
    lin = -nu * k**2
    mask = (np.arange(k.size) < (nx // 3 + 1)).astype(float)   # 2/3 dealias

    def nonlin(vh):
        u = np.fft.irfft(vh, n=nx, axis=1)
        return -0.5j * k * np.fft.rfft(u * u, axis=1) * mask

    n_steps = int(round(t_final / dt))
    E = np.exp(lin * dt / 2)
    E2 = E * E
    vh = np.fft.rfft(u0, axis=1)
    for _ in range(n_steps):
        a = nonlin(vh)
        b = nonlin(E * (vh + dt / 2 * a))
        c = nonlin(E * vh + dt / 2 * b)
        d = nonlin(E2 * vh + dt * E * c)
        vh = E2 * vh + dt / 6 * (E2 * a + 2 * E * (b + c) + d)
    return np.fft.irfft(vh, n=nx, axis=1)


def sanity_checks():
    u0 = random_initial_conditions(4, seed=123)
    uA = solve_burgers(u0, dt=5e-4)
    uB = solve_burgers(u0, dt=2.5e-4)
    print("time-step convergence (max diff dt=5e-4 vs 2.5e-4):", np.abs(uA - uB).max())
    print("mean conserved? |mean(u0)-mean(u1)| max:", np.abs(u0.mean(1) - uA.mean(1)).max())
    print("energy decays? 0.5*mean(u^2) before/after:",
          (0.5 * (u0**2).mean(1)).round(3), (0.5 * (uA**2).mean(1)).round(3))
    # the solver must reduce to the heat equation when the nonlinearity is tiny
    small = 1e-6 * u0
    us = solve_burgers(small, dt=5e-4)
    k = 2 * np.pi * np.fft.rfftfreq(NX, d=1.0 / NX)
    exact = np.fft.irfft(np.fft.rfft(small, axis=1) * np.exp(-NU * k**2 * 1.0), n=NX, axis=1)
    print("linear-limit check vs exact heat solution (relative):",
          np.abs(us - exact).max() / np.abs(small).max())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=200)
    args = p.parse_args()

    sanity_checks()
    if not args.check:
        n = args.n_train + args.n_test
        u0 = random_initial_conditions(n, seed=0)
        u1 = solve_burgers(u0)
        os.makedirs("data", exist_ok=True)
        np.savez("data/burgers_fno.npz",
                 x_train=u0[:args.n_train], y_train=u1[:args.n_train],
                 x_test=u0[args.n_train:], y_test=u1[args.n_train:])
        print(f"saved data/burgers_fno.npz  train={args.n_train} test={args.n_test} nx={NX}")
        print("u0 range:", u0.min().round(2), u0.max().round(2),
              "| u(t=1) range:", u1.min().round(2), u1.max().round(2))
