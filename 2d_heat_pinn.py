import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs("outputs", exist_ok=True)

torch.manual_seed(42)


# problem: 2d heat equation
# du/dt = alpha * (d2u/dx2 + d2u/dy2)
# domain: x,y in [0,1], t in [0,1]
# IC: u(x,y,0) = sin(pi*x) * sin(pi*y)
# BC: u=0 on all 4 walls
# exact: exp(-2*alpha*pi^2*t) * sin(pi*x) * sin(pi*y)

ALPHA = 0.01

def exact(x, y, t):
    return (torch.exp(-2 * ALPHA * np.pi**2 * t)
            * torch.sin(np.pi * x)
            * torch.sin(np.pi * y))


# ---------------------------------------------------------------
# network: input [x, y, t] -> output u
# wider than 1d because 2d is harder to learn
# ---------------------------------------------------------------

class PINN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, y, t):
        return self.net(torch.cat([x, y, t], dim=1))


# ---------------------------------------------------------------
# collocation samplers
# ---------------------------------------------------------------

def interior(N=3000):
    x = torch.rand(N, 1, requires_grad=True)
    y = torch.rand(N, 1, requires_grad=True)
    t = torch.rand(N, 1, requires_grad=True)
    return x, y, t

def ic_points(N=500):
    x = torch.rand(N, 1)
    y = torch.rand(N, 1)
    t = torch.zeros(N, 1)
    u_true = torch.sin(np.pi * x) * torch.sin(np.pi * y)
    return x, y, t, u_true

def bc_points(N=200):
    # 4 walls: x=0, x=1, y=0, y=1
    # u=0 on all of them
    t = torch.rand(N, 1)
    s = torch.rand(N, 1)
    # left/right walls (x fixed, y varies)
    x_l = torch.zeros(N, 1); y_l = s.clone()
    x_r = torch.ones(N, 1);  y_r = s.clone()
    # bottom/top walls (y fixed, x varies)
    x_b = s.clone(); y_b = torch.zeros(N, 1)
    x_t = s.clone(); y_t = torch.ones(N, 1)
    return (x_l, y_l), (x_r, y_r), (x_b, y_b), (x_t, y_t), t


# ---------------------------------------------------------------
# physics loss
# residual = du/dt - alpha*(d2u/dx2 + d2u/dy2)
# laplacian = d2u/dx2 + d2u/dy2
# ---------------------------------------------------------------

def physics_loss(model, x, y, t):
    u = model(x, y, t)

    du_dt = torch.autograd.grad(
        u, t, torch.ones_like(u), create_graph=True)[0]

    du_dx = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]

    du_dy = torch.autograd.grad(
        u, y, torch.ones_like(u), create_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(
        du_dy, y, torch.ones_like(du_dy), create_graph=True)[0]

    laplacian = d2u_dx2 + d2u_dy2
    r = du_dt - ALPHA * laplacian
    return torch.mean(r ** 2)


# ---------------------------------------------------------------
# training
# ---------------------------------------------------------------

model = PINN2D()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

history = []
print(f"{'epoch':<8} {'total':<12} {'pde':<12} {'ic':<12} {'bc':<10}")
print("-" * 54)

for epoch in range(1, 8001):
    opt.zero_grad()

    x_int, y_int, t_int = interior(3000)
    loss_pde = physics_loss(model, x_int, y_int, t_int)

    x_ic, y_ic, t_ic, u_ic = ic_points(500)
    loss_ic = torch.mean((model(x_ic, y_ic, t_ic) - u_ic) ** 2)

    *walls, t_bc = bc_points(200)   # one call: 4 walls + shared t
    loss_bc = sum(torch.mean(model(xb, yb, t_bc) ** 2) for xb, yb in walls)

    loss = loss_pde + 10 * loss_ic + 10 * loss_bc
    loss.backward()
    opt.step()

    history.append(loss.item())
    if epoch % 1000 == 0:
        print(f"{epoch:<8} {loss.item():<12.5f} {loss_pde.item():<12.5f} "
              f"{loss_ic.item():<12.5f} {loss_bc.item():<10.5f}")

print("\ndone.")

# ---------------------------------------------------------------
# evaluate and plot - 2d snapshots at different times
# ---------------------------------------------------------------

model.eval()
n = 50
x1d = np.linspace(0, 1, n)
y1d = np.linspace(0, 1, n)
X2d, Y2d = np.meshgrid(x1d, y1d)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("2D Heat Equation PINN  |  du/dt = alpha*(d2u/dx2 + d2u/dy2)",
             fontsize=13, fontweight='bold')

t_snapshots = [0.0, 0.1, 0.3, 0.6]

for col, ti in enumerate(t_snapshots):
    xf = torch.tensor(X2d.flatten(), dtype=torch.float32).unsqueeze(1)
    yf = torch.tensor(Y2d.flatten(), dtype=torch.float32).unsqueeze(1)
    tf = torch.full((n*n, 1), ti)

    with torch.no_grad():
        u_pred = model(xf, yf, tf).numpy().reshape(n, n)
    u_true = (np.exp(-2*ALPHA*np.pi**2*ti)
              * np.sin(np.pi*X2d) * np.sin(np.pi*Y2d))

    # pinn prediction
    ax = axes[0, col]
    im = ax.contourf(X2d, Y2d, u_pred, levels=30, cmap='hot', vmin=0)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"PINN  t={ti}")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # exact solution
    ax = axes[1, col]
    im2 = ax.contourf(X2d, Y2d, u_true, levels=30, cmap='hot', vmin=0)
    plt.colorbar(im2, ax=ax)
    ax.set_title(f"Exact  t={ti}")
    ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("outputs/day3_result.png", dpi=150, bbox_inches='tight')
print("plot saved to outputs/day3_result.png")

# accuracy at several times (rel L2 vs exact solution)
xf = torch.tensor(X2d.flatten(), dtype=torch.float32).unsqueeze(1)
yf = torch.tensor(Y2d.flatten(), dtype=torch.float32).unsqueeze(1)
print()
for ti in [0.0, 0.1, 0.3, 0.6, 1.0]:
    tf = torch.full((n*n, 1), ti)
    with torch.no_grad():
        u_p = model(xf, yf, tf).numpy().reshape(n, n)
    u_t = (np.exp(-2*ALPHA*np.pi**2*ti) * np.sin(np.pi*X2d) * np.sin(np.pi*Y2d))
    e = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
    print(f"relative L2 error at t={ti:<4}: {e*100:.3f}%   max abs err: {np.abs(u_p-u_t).max():.5f}")

import os
os.makedirs("outputs", exist_ok=True)
