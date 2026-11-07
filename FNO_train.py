"""
 Fourier Neural Operator (FNO) and Physics-Informed FNO (PINO) for Burgers

Learns the operator   u(x,0) -> u(x,1)   for periodic 1D Burgers (nu = 0.01),
data from FNO_dataset.py.

The network is a function of the initial condition AND a time t:
        u(x,t) = u0(x) + t * g(u0, x, t)        (so u(x,0) = u0 exactly)

losses
  data    : relative L2 between u(x,1) and the solver solution, on the first
            --n-labeled training samples only
  physics : PDE residual  u_t + u u_x - nu u_xx  at random times t in (0,1),
            on ALL training initial conditions (no labels needed).
            u_x, u_xx by FFT (exact for periodic data), u_t by central
            finite difference in t.
            Switched on with --physics-weight > 0   (that is the PINO part)

usage
  python fno_fno_train.py                                    # plain FNO, all labels
  python fno_fno_train.py --n-labeled 50                     # FNO, few labels
  python fno_fno_train.py --n-labeled 50 --physics-weight 0.01   # PINO, few labels
"""
import argparse
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--n-labeled', type=int, default=1000)
p.add_argument('--physics-weight', type=float, default=0.0)
p.add_argument('--epochs', type=int, default=300)
p.add_argument('--batch', type=int, default=50)
p.add_argument('--modes', type=int, default=16)
p.add_argument('--width', type=int, default=32)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--dealias', action='store_true',
               help='low-pass filter (2/3 rule) inside the physics loss, like the solver')
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.makedirs('outputs', exist_ok=True)
NU, H_T = 0.01, 0.01                      # viscosity, finite-difference step in t

# ---------------- data ----------------
d = np.load('data/burgers_fno.npz')
x_tr = torch.tensor(d['x_train'], dtype=torch.float32)
y_tr = torch.tensor(d['y_train'], dtype=torch.float32)
x_te = torch.tensor(d['x_test'], dtype=torch.float32)
y_te = torch.tensor(d['y_test'], dtype=torch.float32)
N_X = x_tr.shape[1]
n_lab = min(args.n_labeled, x_tr.shape[0])
x_lab, y_lab = x_tr[:n_lab], y_tr[:n_lab]

K = 2 * np.pi * torch.fft.rfftfreq(N_X, d=1.0 / N_X)          # wavenumbers on [0,1)


# ---------------- model ----------------
class SpectralConv1d(nn.Module):
    """multiply the lowest `modes` Fourier coefficients by learned complex weights"""
    def __init__(self, c_in, c_out, modes):
        super().__init__()
        self.c_out, self.modes = c_out, modes
        self.w = nn.Parameter(torch.randn(c_in, c_out, modes, 2) / (c_in * c_out))

    def forward(self, x):                                      # x: (B, C, N)
        B, _, N = x.shape
        xh = torch.fft.rfft(x, dim=-1)                         # (B, C, N//2+1)
        w = torch.view_as_complex(self.w)                      # (c_in, c_out, modes)
        out = torch.zeros(B, self.c_out, N // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes] = torch.einsum('bix,iox->box', xh[:, :, :self.modes], w)
        return torch.fft.irfft(out, n=N, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, modes, width, n_layers=4):
        super().__init__()
        self.lift = nn.Linear(3, width)                        # inputs: u0, x, t
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.skip = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))
        self.act = nn.GELU()

    def forward(self, u0, t):                                  # u0: (B,N)  t: (B,)
        B, N = u0.shape
        grid = torch.linspace(0, 1, N).unsqueeze(0).expand(B, N)
        tt = t.unsqueeze(1).expand(B, N)
        h = self.lift(torch.stack([u0, grid, tt], dim=-1)).permute(0, 2, 1)   # (B,W,N)
        for spec, skip in zip(self.spec, self.skip):
            h = self.act(spec(h) + skip(h))
        g = self.proj(h.permute(0, 2, 1)).squeeze(-1)                          # (B,N)
        return u0 + t.unsqueeze(1) * g                                         # u(x,0)=u0


def rel_l2(pred, target):
    """mean over samples of ||pred-target|| / ||target||"""
    return (torch.linalg.norm(pred - target, dim=1) / torch.linalg.norm(target, dim=1)).mean()


MASK = (torch.arange(N_X // 2 + 1) < (N_X // 3 + 1)).float()   # 2/3 dealiasing mask


def physics_loss(model, u0):
    """PDE residual at random times, derivatives: FFT in x, central difference in t"""
    B = u0.shape[0]
    t = torch.rand(B) * (1 - 2 * H_T) + H_T
    u = model(u0, t)
    u_t = (model(u0, t + H_T) - model(u0, t - H_T)) / (2 * H_T)
    uh = torch.fft.rfft(u, dim=1)
    if args.dealias:                       # drop the high modes the solver never had
        uh = uh * MASK
        u = torch.fft.irfft(uh, n=N_X, dim=1)
        u_t = torch.fft.irfft(torch.fft.rfft(u_t, dim=1) * MASK, n=N_X, dim=1)
    u_x = torch.fft.irfft(1j * K * uh, n=N_X, dim=1)
    u_xx = torch.fft.irfft(-(K ** 2) * uh, n=N_X, dim=1)
    return torch.mean((u_t + u * u_x - NU * u_xx) ** 2)


# ---------------- training ----------------
model = FNO1d(args.modes, args.width)
n_params = sum(p.numel() for p in model.parameters())
use_phys = args.physics_weight > 0
name = ('PINO' if use_phys else 'FNO')
tag = f"{name}{'_w' + str(args.physics_weight) if use_phys else ''}{'_dealias' if args.dealias else ''}_lab{n_lab}_seed{args.seed}"
print(f"{tag} | params: {n_params:,} | labeled: {n_lab}/{x_tr.shape[0]} | "
      f"physics weight: {args.physics_weight}")

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
steps_per_epoch = max(1, x_tr.shape[0] // args.batch)   # same number of updates for every setup


def test_error():
    model.eval()
    with torch.no_grad():
        pred = model(x_te, torch.ones(x_te.shape[0]))
    model.train()
    return rel_l2(pred, y_te).item(), pred


for epoch in range(1, args.epochs + 1):
    d_sum = p_sum = 0.0
    for _ in range(steps_per_epoch):
        opt.zero_grad()
        i = torch.randint(0, n_lab, (min(args.batch, n_lab),))
        l_data = rel_l2(model(x_lab[i], torch.ones(i.numel())), y_lab[i])
        loss = l_data
        if use_phys:
            j = torch.randint(0, x_tr.shape[0], (args.batch,))
            l_phys = physics_loss(model, x_tr[j])
            loss = loss + args.physics_weight * l_phys
            p_sum += l_phys.item()
        loss.backward()
        opt.step()
        d_sum += l_data.item()
    sched.step()
    if epoch % 50 == 0 or epoch == 1:
        te, _ = test_error()
        print(f"epoch {epoch:<4} data(rel L2) {d_sum/steps_per_epoch:.4f}  "
              f"physics {p_sum/steps_per_epoch:.5f}  test rel L2 {te:.4f}")

te, pred = test_error()
print(f"\nFINAL {tag}: test relative L2 error = {te*100:.2f}%   "
      f"(predicting zeros would be 100%)")

# ---------------- save results ----------------
csv_path = 'outputs/fno_results.csv'
new = not os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as f:
    w = csv.writer(f)
    if new:
        w.writerow(['model', 'n_labeled', 'physics_weight', 'seed', 'epochs', 'test_rel_l2_pct'])
    w.writerow([name, n_lab, args.physics_weight, args.seed, args.epochs, f"{te*100:.2f}"])

x = np.linspace(0, 1, N_X, endpoint=False)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, k in zip(axes, [0, 1, 2]):
    ax.plot(x, x_te[k].numpy(), color='gray', alpha=0.5, label='u(x,0) input')
    ax.plot(x, y_te[k].numpy(), 'k--', lw=2, label='solver u(x,1)')
    ax.plot(x, pred[k].numpy(), 'r-', lw=1.5, label=f'{name} u(x,1)')
    ax.set_title(f"test sample {k}"); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
plt.suptitle(f"{tag}: test rel L2 = {te*100:.2f}%")
plt.tight_layout()
plt.savefig(f"outputs/{tag}.png", dpi=150, bbox_inches='tight')
print(f"plot saved to outputs/{tag}.png")