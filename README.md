# PINN-Project: physics-informed learning for heat and Burgers equations

I built this to learn how neural networks can solve PDEs, starting from autograd basics and ending with operator learning. Everything runs on a laptop CPU in PyTorch.

The project has three parts:

1. **PINNs** for the 1D heat, 2D heat and Burgers equations
2. **L-BFGS fine-tuning**, which fixed most of the error on Burgers
3. **FNO and PINO** (Fourier neural operators), compared on how few labeled samples they need

## Results

| Problem | Method | Relative L2 error vs exact |
|---|---|---|
| 1D heat | PINN | TODO (paste exact value from `1d_heat_pinn.py`) |
| 2D heat | PINN | 0.90% at t=0, 1.12% at t=1 (8000 epochs) |
| Burgers | PINN, Adam 10k epochs | 5.70% (max abs error 0.605) |
| Burgers | Same network, no physics (IC/BC data only) | 48.63% |
| Burgers | Adam 5k epochs only | ~32% (seed 42) |
| Burgers | Adam 5k + L-BFGS | ~0.5% (seed 42) |

The exact solutions are the analytical solution for the heat equations and the Cole-Hopf solution for Burgers (`utils/burgers_reference.py`).

![burgers](outputs/burgers_result.png)

### L-BFGS fine-tuning (Burgers)

Adam alone stalls at about 32% error. Switching to L-BFGS on a fixed set of collocation points brought it down to about 0.5% on seed 42.

| Seed | Adam only | Adam + L-BFGS |
|---|---|---|
| 42 | TODO | TODO |
| 1 | TODO | TODO |
| 2 | TODO | TODO |

I also tried putting half the collocation points near the shock (`--refine`). In my run this made things worse (3.36% vs about 0.5%, seed 42). I only ran it once, so I can't say why.

### FNO vs PINO (Burgers, periodic, nu = 0.01)

Both models learn the map from an initial condition u(x,0) to the solution u(x,1), trained on 1000 random initial conditions with solver data from a pseudo-spectral solver (`FNO_dataset.py`). The PINO adds the PDE residual as an extra loss, using all 1000 initial conditions without needing labels. I varied how many of them have labels.

| Labeled samples | FNO | PINO (weight 0.1, filtered) |
|---|---|---|
| 1000 | 4.38% | TODO |
| 50 | 49.84%, 49.87%, 53.72% (3 seeds) | TODO seed 0, 36.99%, 37.57% |

The physics loss helps a lot when labels are scarce. The first PINO version (no filtering, weight 0.01) was worse than the plain FNO with all 1000 labels (7.54% vs 4.38%).

## What I learned and what went wrong

- **Boundary sampler bug.** In the 2D heat script, boundary points were sampled three times per epoch with mismatched times. I fixed it to sample once.
- **The shock is hard.** In Burgers the error concentrates in a thin line at x=0 and grows with time (3.6% at t=0.25, 11.4% at t=1.0).
- **No-physics baseline.** The same network trained only on initial and boundary data fits them (loss 0.001) but has 48.6% error inside the domain, which is the point of adding the PDE loss.
- **PINO needed a filter.** My guess is that the spectral derivative amplifies high-frequency noise in the network output (by up to about k^2). Filtering with the same 2/3 rule the solver uses brought the 50-label error down further. I didn't prove that this is the cause.
- **FNO viscosity.** My first dataset used nu = 0.1, but everything had decayed to nearly zero by t=1, which made the problem trivial. I switched to nu = 0.01.

## Limitations

- Seeds: 3 seeds for the FNO/PINO experiments and L-BFGS, 1 for everything else. These are small samples.
- I picked the PINO weight (0.1) by looking at test error, so that number is slightly optimistic.
- PINO is slower per epoch than the FNO because of the extra forward passes for the time derivative.
- Even with the physics loss, the PINO residual stays around 1, so it doesn't fully satisfy the PDE.

## Running it

```bash
pip install -r requirements.txt
python 1d_heat_pinn.py
python 2d_heat_pinn.py
python burgers_pinn.py
python lbfgs_finetune.py --seed 42
python FNO_dataset.py
python FNO_train.py --n-labeled 50
python FNO_train.py --n-labeled 50 --physics-weight 0.1 --dealias
```

Results are saved to `outputs/` (FNO/PINO runs also append to `outputs/fno_results.csv`).

## Files

```
autograd_basics.py   gradients of network outputs w.r.t. inputs
1d_heat_pinn.py      1D heat equation PINN
2d_heat_pinn.py      2D heat equation PINN
burgers_pinn.py      Burgers PINN + no-physics baseline
lbfgs_finetune.py    Adam -> L-BFGS for Burgers
FNO_dataset.py       spectral solver + dataset for the FNO
FNO_train.py         FNO and PINO training
utils/               exact Burgers solution (Cole-Hopf)
outputs/             plots and fno_results.csv
```

## Tools

I used Claude (an AI assistant) to help write and debug code. I ran every experiment myself, and all numbers above come from my own runs.