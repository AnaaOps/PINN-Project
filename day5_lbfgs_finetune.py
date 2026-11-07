"""
Day 5 - improving the Burgers PINN: Adam -> L-BFGS fine-tuning
(+ optional extra collocation points near the shock)

usage:
    python day5_lbfgs_finetune.py              # Adam then L-BFGS, uniform points
    python day5_lbfgs_finetune.py --refine     # same, but half the points near x=0

prints the relative L2 error vs the exact Cole-Hopf solution after each stage,
so the effect of each technique is measured separately.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.burgers_reference import burgers_grid

parser = argparse.ArgumentParser()
parser.add_argument('--refine', action='store_true',
                    help='put half of the collocation points near the shock (|x|<0.2)')
parser.add_argument('--adam-epochs', type=int, default=5000)
parser.add_argument('--lbfgs-iters', type=int, default=1500)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

os.makedirs('outputs', exist_ok=True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
NU = 0.01 / np.pi
TAG = ('refine' if args.refine else 'uniform') + f'_seed{args.seed}'


class BurgersPINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, 40), nn.Tanh()]
        for _ in range(4):
            layers += [nn.Linear(40, 40), nn.Tanh()]
        layers += [nn.Linear(40, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def interior(N):
    """collocation points; with --refine half are concentrated near x=0"""
    if args.refine:
        n_near = N // 2
        x = torch.cat([torch.rand(N - n_near, 1) * 2 - 1,
                       (torch.rand(n_near, 1) * 2 - 1) * 0.2])
    else:
        x = torch.rand(N, 1) * 2 - 1
    t = torch.rand(N, 1)
    return x.requires_grad_(True), t.requires_grad_(True)


def ic_points(N):
    x = torch.rand(N, 1) * 2 - 1
    return x, torch.zeros(N, 1), -torch.sin(np.pi * x)


def bc_points(N):
    t = torch.rand(N, 1)
    return -torch.ones(N, 1), torch.ones(N, 1), t


def physics_loss(model, x, t):
    u = model(x, t)
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]
    return torch.mean((du_dt + u * du_dx - NU * d2u_dx2) ** 2)


def total_loss(model, x_int, t_int, ic, bc):
    x_ic, t_ic, u_ic = ic
    x_l, x_r, t_bc = bc
    l_pde = physics_loss(model, x_int, t_int)
    l_ic = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
    l_bc = torch.mean(model(x_l, t_bc) ** 2) + torch.mean(model(x_r, t_bc) ** 2)
    return l_pde + 10 * l_ic + 10 * l_bc


# ---- exact reference for scoring -------------------------------------
x_ref, t_ref, U_ref = burgers_grid(nx=256, nt=101)
Xg, Tg = np.meshgrid(x_ref, t_ref, indexing='ij')
xf = torch.tensor(Xg.flatten(), dtype=torch.float32).unsqueeze(1)
tf = torch.tensor(Tg.flatten(), dtype=torch.float32).unsqueeze(1)


def evaluate(model, label):
    model.eval()
    with torch.no_grad():
        U = model(xf, tf).numpy().reshape(Xg.shape)
    model.train()
    err = np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref)
    print(f"[{label}] relative L2 error vs exact: {err*100:.2f}%   "
          f"max abs error: {np.abs(U - U_ref).max():.4f}")
    return U, err


model = BurgersPINN()

# ---- stage 1: Adam, fresh points every epoch -------------------------
print(f"mode: {TAG} | Adam epochs: {args.adam_epochs} | L-BFGS iters: {args.lbfgs_iters}")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(1, args.adam_epochs + 1):
    opt.zero_grad()
    loss = total_loss(model, *interior(3000), ic_points(200), bc_points(100))
    loss.backward()
    opt.step()
    if epoch % 1000 == 0:
        print(f"  adam {epoch:<6} loss {loss.item():.5f}")
U_adam, err_adam = evaluate(model, 'after Adam')

# ---- stage 2: L-BFGS on a FIXED set of points ------------------------
# L-BFGS needs a deterministic loss, so the points are sampled once.
x_fix, t_fix = interior(10000)
ic_fix, bc_fix = ic_points(400), bc_points(200)

lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=args.lbfgs_iters,
                          max_eval=int(args.lbfgs_iters * 1.25), history_size=50,
                          tolerance_grad=1e-9, tolerance_change=1e-12,
                          line_search_fn='strong_wolfe')
step_count = [0]


def closure():
    lbfgs.zero_grad()
    loss = total_loss(model, x_fix, t_fix, ic_fix, bc_fix)
    loss.backward()
    step_count[0] += 1
    if step_count[0] % 100 == 0:
        print(f"  lbfgs eval {step_count[0]:<5} loss {loss.item():.6f}")
    return loss


lbfgs.step(closure)
U_lb, err_lb = evaluate(model, 'after L-BFGS')

print(f"\nsummary ({TAG}): Adam {err_adam*100:.2f}%  ->  Adam + L-BFGS {err_lb*100:.2f}%")

# ---- plot ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
j = 75
axes[0].plot(x_ref, U_ref[:, j], 'k--', lw=2, label='exact')
axes[0].plot(x_ref, U_adam[:, j], 'b-', alpha=0.7, label=f'Adam ({err_adam*100:.1f}%)')
axes[0].plot(x_ref, U_lb[:, j], 'r-', lw=2, label=f'Adam+L-BFGS ({err_lb*100:.1f}%)')
axes[0].set_title('t = 0.75'); axes[0].legend(); axes[0].grid(alpha=0.3)
for ax, U, name in [(axes[1], U_adam, 'Adam'), (axes[2], U_lb, 'Adam + L-BFGS')]:
    im = ax.contourf(Tg, Xg, np.abs(U - U_ref), levels=50, cmap='RdYlGn_r', vmin=0, vmax=0.6)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'|error| - {name}'); ax.set_xlabel('t'); ax.set_ylabel('x')
plt.suptitle(f'Burgers PINN: L-BFGS fine-tuning ({TAG} collocation points)')
plt.tight_layout()
out = f'outputs/day5_lbfgs_{TAG}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('plot saved to', out)