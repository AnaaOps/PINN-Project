import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ---------------------------------------------------------------
# problem: 1d heat equation
# du/dt = alpha * d2u/dx2
# domain: x in [0,1], t in [0,1]
# IC: u(x,0) = sin(pi*x)
# BC: u(0,t) = 0, u(1,t) = 0
# exact solution: exp(-alpha*pi^2*t) * sin(pi*x)
# ---------------------------------------------------------------

ALPHA = 0.01

def exact(x, t):
    return torch.exp(-ALPHA * np.pi**2 * t) * torch.sin(np.pi * x)


# ---------------------------------------------------------------
# network: input [x,t] -> output u
# tanh activation - relu breaks the 2nd derivative completely
# ---------------------------------------------------------------

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),  nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# ---------------------------------------------------------------
# collocation point samplers
# new random points each epoch - not a fixed dataset
# ---------------------------------------------------------------

def interior(N=2000):
    # requires_grad=True on both - need du/dx and du/dt
    x = torch.rand(N, 1, requires_grad=True)
    t = torch.rand(N, 1, requires_grad=True)
    return x, t

def ic_points(N=200):
    x = torch.rand(N, 1)
    t = torch.zeros(N, 1)
    u_true = torch.sin(np.pi * x)
    return x, t, u_true

def bc_points(N=100):
    t = torch.rand(N, 1)
    x0 = torch.zeros(N, 1)   # left wall
    x1 = torch.ones(N, 1)    # right wall
    return x0, x1, t


# ---------------------------------------------------------------
# physics loss - core pinn mechanic
# residual r = du/dt - alpha * d2u/dx2
# want r -> 0 everywhere in domain
# ---------------------------------------------------------------

def physics_loss(model, x, t):
    u = model(x, t)

    du_dt = torch.autograd.grad(
        u, t, torch.ones_like(u), create_graph=True)[0]

    du_dx = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True)[0]

    d2u_dx2 = torch.autograd.grad(
        du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]

    r = du_dt - ALPHA * d2u_dx2
    return torch.mean(r ** 2)


# ---------------------------------------------------------------
# training
# total loss = pde + 10*ic + 10*bc
# ic/bc weighted higher because they anchor the whole solution
# ---------------------------------------------------------------

model = PINN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

history = []
print(f"{'epoch':<8} {'total':<12} {'pde':<12} {'ic':<12} {'bc':<10}")
print("-" * 54)

for epoch in range(1, 5001):
    opt.zero_grad()

    x_int, t_int = interior(2000)
    loss_pde = physics_loss(model, x_int, t_int)

    x_ic, t_ic, u_ic = ic_points(200)
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)

    x0, x1, t_bc = bc_points(100)
    loss_bc = torch.mean(model(x0, t_bc) ** 2) + torch.mean(model(x1, t_bc) ** 2)

    loss = loss_pde + 10 * loss_ic + 10 * loss_bc
    loss.backward()
    opt.step()

    history.append(loss.item())
    if epoch % 500 == 0:
        print(f"{epoch:<8} {loss.item():<12.5f} {loss_pde.item():<12.5f} "
              f"{loss_ic.item():<12.5f} {loss_bc.item():<10.5f}")

print("\ndone.")

# ---------------------------------------------------------------
# evaluate and plot
# ---------------------------------------------------------------

model.eval()
x_plot = torch.linspace(0, 1, 100).unsqueeze(1)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("1D Heat Equation PINN  |  du/dt = alpha * d2u/dx2", fontsize=13, fontweight='bold')

colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a']
t_slices = [0.0, 0.2, 0.5, 1.0]

# time slices: pinn vs exact
ax = axes[0, 0]
for ti, col in zip(t_slices, colors):
    t_p = torch.full((100, 1), ti)
    with torch.no_grad():
        u_pred = model(x_plot, t_p).numpy()
    u_true = exact(x_plot, t_p).numpy()
    ax.plot(x_plot.numpy(), u_true, '--', color=col, alpha=0.5, lw=2)
    ax.plot(x_plot.numpy(), u_pred, '-',  color=col, lw=2, label=f't={ti}')
ax.set_xlabel("x"); ax.set_ylabel("u")
ax.set_title("PINN (solid) vs exact (dashed)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# pinn heatmap
ax = axes[0, 1]
T, X = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
xf = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
tf = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    uf = model(xf, tf).numpy().reshape(100, 100)
im = ax.contourf(T, X, uf, levels=50, cmap='hot')
plt.colorbar(im, ax=ax)
ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title("PINN prediction")

# exact heatmap
ax = axes[0, 2]
u_ex = (torch.exp(torch.tensor(-ALPHA * np.pi**2 * T)) *
        torch.sin(torch.tensor(np.pi * X))).numpy()
im2 = ax.contourf(T, X, u_ex, levels=50, cmap='hot')
plt.colorbar(im2, ax=ax)
ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title("Exact solution")

# error heatmap
ax = axes[1, 0]
err = np.abs(uf - u_ex)
im3 = ax.contourf(T, X, err, levels=50, cmap='RdYlGn_r')
plt.colorbar(im3, ax=ax)
ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(f"Absolute error (max={err.max():.4f})")

# loss curve
ax = axes[1, 1]
ax.semilogy(history, color='#e63946', lw=1.5)
ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)"); ax.set_title("Training loss")
ax.grid(alpha=0.3)

# final time slice
ax = axes[1, 2]
t_end = torch.ones(100, 1)
with torch.no_grad():
    u_end = model(x_plot, t_end).numpy()
u_ex_end = exact(x_plot, t_end).numpy()
ax.plot(x_plot.numpy(), u_ex_end, 'k--', lw=2, label='exact')
ax.plot(x_plot.numpy(), u_end,    'r-',  lw=2, label='PINN')
ax.fill_between(x_plot.numpy().flatten(), u_end.flatten(), u_ex_end.flatten(),
                alpha=0.25, color='red')
ax.set_xlabel("x"); ax.set_ylabel("u at t=1"); ax.set_title("t=1.0 comparison")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/heat_1d_result.png", dpi=150, bbox_inches='tight')
print("plot saved to outputs/heat_1d_result.png")

rel_err = np.linalg.norm(uf - u_ex) / np.linalg.norm(u_ex)
print(f"\nrelative L2 error : {rel_err*100:.3f}%")
print(f"max absolute error: {err.max():.5f}")
