import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from utils.burgers_reference import burgers_grid

os.makedirs('outputs', exist_ok=True)

torch.manual_seed(42)

# ---------------------------------------------------------------
# problem: burgers equation (nonlinear - the real benchmark)
# du/dt + u*du/dx = nu * d2u/dx2
#
# u*du/dx is the nonlinear convection term - this is what makes
# it hard. creates a shock wave near x=0 around t=0.5-0.7
#
# this is the exact problem used in the original pinn paper
# by raissi et al. 2019
#
# domain: x in [-1,1], t in [0,1]
# IC: u(x,0) = -sin(pi*x)
# BC: u(-1,t) = 0, u(1,t) = 0
# nu = 0.01/pi  (standard from the paper)
# ---------------------------------------------------------------

NU = 0.01 / np.pi


# ---------------------------------------------------------------
# network - same structure as 1d heat but larger
# needs more capacity to capture the shock
# ---------------------------------------------------------------

class BurgersPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 40),  nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# ---------------------------------------------------------------
# samplers - x in [-1,1] not [0,1]
# ---------------------------------------------------------------

def interior(N=3000):
    # x in [-1, 1]
    x = (torch.rand(N, 1) * 2 - 1).requires_grad_(True)
    t = torch.rand(N, 1).requires_grad_(True)
    return x, t

def ic_points(N=200):
    x = torch.rand(N, 1) * 2 - 1
    t = torch.zeros(N, 1)
    u_true = -torch.sin(np.pi * x)   # IC: u(x,0) = -sin(pi*x)
    return x, t, u_true

def bc_points(N=100):
    t = torch.rand(N, 1)
    x_left  = -torch.ones(N, 1)
    x_right =  torch.ones(N, 1)
    return x_left, x_right, t


# ---------------------------------------------------------------
# physics loss - burgers residual
# r = du/dt + u*du/dx - nu*d2u/dx2
# the u*du/dx term is the nonlinear part
# ---------------------------------------------------------------

def physics_loss(model, x, t):
    u = model(x, t)

    du_dt = torch.autograd.grad(
        u, t, torch.ones_like(u), create_graph=True)[0]

    du_dx = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True)[0]

    d2u_dx2 = torch.autograd.grad(
        du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]

    # full burgers residual - note the u*du_dx nonlinear term
    r = du_dt + u * du_dx - NU * d2u_dx2
    return torch.mean(r ** 2)


# ---------------------------------------------------------------
# training - more epochs needed for the shock
# ---------------------------------------------------------------

model = BurgersPINN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

history = []
print(f"{'epoch':<8} {'total':<12} {'pde':<12} {'ic':<12} {'bc':<10}")
print("-" * 54)

for epoch in range(1, 10001):
    opt.zero_grad()

    x_int, t_int = interior(3000)
    loss_pde = physics_loss(model, x_int, t_int)

    x_ic, t_ic, u_ic = ic_points(200)
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)

    x_l, x_r, t_bc = bc_points(100)
    loss_bc = (torch.mean(model(x_l, t_bc) ** 2) +
               torch.mean(model(x_r, t_bc) ** 2))

    loss = loss_pde + 10 * loss_ic + 10 * loss_bc
    loss.backward()
    opt.step()

    history.append(loss.item())
    if epoch % 1000 == 0:
        print(f"{epoch:<8} {loss.item():<12.5f} {loss_pde.item():<12.5f} "
              f"{loss_ic.item():<12.5f} {loss_bc.item():<10.5f}")

print("\ndone.")

# ---------------------------------------------------------------
# baseline: same network, NO physics - trained only on IC + BC data
# shows what a normal nn does when it has no PDE to enforce
# ---------------------------------------------------------------

torch.manual_seed(42)
baseline = BurgersPINN()
opt_b = torch.optim.Adam(baseline.parameters(), lr=1e-3)
for epoch in range(1, 10001):
    opt_b.zero_grad()
    x_ic, t_ic, u_ic = ic_points(200)
    x_l, x_r, t_bc = bc_points(100)
    loss_b = (torch.mean((baseline(x_ic, t_ic) - u_ic) ** 2)
              + torch.mean(baseline(x_l, t_bc) ** 2)
              + torch.mean(baseline(x_r, t_bc) ** 2))
    loss_b.backward()
    opt_b.step()
print(f"baseline (no physics) final data loss: {loss_b.item():.6f}")

# ---------------------------------------------------------------
# evaluate against the exact (Cole-Hopf) reference solution
# ---------------------------------------------------------------

model.eval(); baseline.eval()
x_ref, t_ref, U_ref = burgers_grid(nx=256, nt=101)      # U_ref: (nx, nt)
Xg, Tg = np.meshgrid(x_ref, t_ref, indexing='ij')
xf = torch.tensor(Xg.flatten(), dtype=torch.float32).unsqueeze(1)
tf = torch.tensor(Tg.flatten(), dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    U_pinn = model(xf, tf).numpy().reshape(Xg.shape)
    U_base = baseline(xf, tf).numpy().reshape(Xg.shape)

def rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

err_pinn = rel_l2(U_pinn, U_ref)
err_base = rel_l2(U_base, U_ref)
print(f"\nPINN     relative L2 error vs exact: {err_pinn*100:.2f}%")
print(f"baseline relative L2 error vs exact: {err_base*100:.2f}%")
print(f"PINN max abs error: {np.abs(U_pinn - U_ref).max():.4f}")
for ti in [0.25, 0.5, 0.75, 1.0]:
    j = int(round(ti * 100))
    print(f"  t={ti:<4} PINN rel L2: {rel_l2(U_pinn[:, j], U_ref[:, j])*100:6.2f}%   "
          f"baseline: {rel_l2(U_base[:, j], U_ref[:, j])*100:6.2f}%")

# ---------------------------------------------------------------
# plots
# ---------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Burgers Equation PINN  |  du/dt + u*du/dx = nu*d2u/dx2\n"
             f"PINN rel L2 error = {err_pinn*100:.2f}%   vs   no-physics baseline = {err_base*100:.2f}%",
             fontsize=12, fontweight='bold')

# snapshots: pinn vs exact
ax = axes[0, 0]
cols = ['#1a1a2e', '#0f3460', '#2a9d8f', '#e94560', '#ff0000']
for ti, col in zip([0.0, 0.25, 0.5, 0.75, 1.0], cols):
    j = int(round(ti * 100))
    ax.plot(x_ref, U_ref[:, j], '--', color=col, alpha=0.6, lw=1.5)
    ax.plot(x_ref, U_pinn[:, j], '-', color=col, lw=2, label=f't={ti}')
ax.set_title("PINN (solid) vs exact (dashed)")
ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# pinn heatmap
ax = axes[0, 1]
im = ax.contourf(Tg, Xg, U_pinn, levels=60, cmap='RdBu_r')
plt.colorbar(im, ax=ax); ax.set_title("PINN u(x,t)")
ax.set_xlabel("t"); ax.set_ylabel("x")

# exact heatmap
ax = axes[0, 2]
im = ax.contourf(Tg, Xg, U_ref, levels=60, cmap='RdBu_r')
plt.colorbar(im, ax=ax); ax.set_title("Exact u(x,t) (Cole-Hopf)")
ax.set_xlabel("t"); ax.set_ylabel("x")

# error heatmap
ax = axes[1, 0]
im = ax.contourf(Tg, Xg, np.abs(U_pinn - U_ref), levels=50, cmap='RdYlGn_r')
plt.colorbar(im, ax=ax); ax.set_title(f"PINN |error| (max={np.abs(U_pinn-U_ref).max():.3f})")
ax.set_xlabel("t"); ax.set_ylabel("x")

# pinn vs baseline at shock time
ax = axes[1, 1]
j = 75
ax.plot(x_ref, U_ref[:, j], 'k--', lw=2, label='exact')
ax.plot(x_ref, U_pinn[:, j], 'r-', lw=2, label='PINN')
ax.plot(x_ref, U_base[:, j], 'b-', lw=1.5, alpha=0.7, label='no-physics NN')
ax.set_title("t=0.75: PINN vs baseline vs exact")
ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# loss
ax = axes[1, 2]
ax.semilogy(history, color='#e63946', lw=1.5)
ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)"); ax.set_title("Training loss")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/day4_result.png", dpi=150, bbox_inches='tight')
print("\nplot saved to outputs/day4_result.png")
