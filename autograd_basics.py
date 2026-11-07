import torch
import torch.nn as nn

# quick check - make sure autograd works on inputs not just weights
# this is the only new thing compared to normal pytorch training

torch.manual_seed(0)

# -----------------------------------------------------------
# part 1: simple function, verify against known answer
# -----------------------------------------------------------

x = torch.tensor([2.0], requires_grad=True)
u = x ** 3   # u = x^3,  du/dx = 3x^2,  d2u/dx2 = 6x

# first derivative
du_dx = torch.autograd.grad(
    u, x,
    grad_outputs=torch.ones_like(u),
    create_graph=True   # keep graph alive, need it for second derivative
)[0]

# second derivative - just call grad again on du_dx
d2u_dx2 = torch.autograd.grad(
    du_dx, x,
    grad_outputs=torch.ones_like(du_dx),
    create_graph=True
)[0]

print("=== part 1: x^3 ===")
print(f"x        = {x.item()}")
print(f"u = x^3  = {u.item()}")
print(f"du/dx    = {du_dx.item():.1f}   (expected {3*x.item()**2:.1f})")
print(f"d2u/dx2  = {d2u_dx2.item():.1f}   (expected {6*x.item():.1f})")

# -----------------------------------------------------------
# part 2: same trick on a real network
# tanh not relu - relu's 2nd derivative is 0 everywhere,
# which kills the physics loss completely
# -----------------------------------------------------------

net = nn.Sequential(
    nn.Linear(1, 16), nn.Tanh(),
    nn.Linear(16, 16), nn.Tanh(),
    nn.Linear(16, 1)
)

x2 = torch.tensor([[1.5]], requires_grad=True)
u2 = net(x2)

du2_dx = torch.autograd.grad(
    u2, x2,
    grad_outputs=torch.ones_like(u2),
    create_graph=True
)[0]

d2u2_dx2 = torch.autograd.grad(
    du2_dx, x2,
    grad_outputs=torch.ones_like(du2_dx),
    create_graph=True
)[0]

print("\n=== part 2: actual network ===")
print(f"u(1.5)   = {u2.item():.4f}  (random, network untrained)")
print(f"du/dx    = {du2_dx.item():.4f}")
print(f"d2u/dx2  = {d2u2_dx2.item():.4f}")
print("numbers don't mean anything yet - point is it works")

# -----------------------------------------------------------
# part 3: two inputs x and t - what pinn actually uses
# get du/dt and d2u/dx2 separately from one forward pass
# -----------------------------------------------------------

net2 = nn.Sequential(
    nn.Linear(2, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.Tanh(),
    nn.Linear(32, 1)
)

# single point: x=0.5, t=0.1
xt = torch.tensor([[0.5, 0.1]], requires_grad=True)
u3 = net2(xt)

# grad w.r.t both inputs at once - shape (1, 2) = [du/dx, du/dt]
grads = torch.autograd.grad(
    u3, xt,
    grad_outputs=torch.ones_like(u3),
    create_graph=True
)[0]

du_dx3 = grads[:, 0:1]
du_dt3 = grads[:, 1:2]

# second derivative w.r.t x only
d2u_dx2_3 = torch.autograd.grad(
    du_dx3, xt,
    grad_outputs=torch.ones_like(du_dx3),
    create_graph=True
)[0][:, 0:1]

ALPHA = 0.01
residual = du_dt3 - ALPHA * d2u_dx2_3

print("\n=== part 3: two inputs [x, t] ===")
print(f"u(0.5, 0.1)  = {u3.item():.4f}")
print(f"du/dx        = {du_dx3.item():.4f}")
print(f"du/dt        = {du_dt3.item():.4f}")
print(f"d2u/dx2      = {d2u_dx2_3.item():.4f}")
print(f"\nheat eq residual = du/dt - alpha*d2u/dx2 = {residual.item():.4f}")
print("(after training this should be close to 0)")
