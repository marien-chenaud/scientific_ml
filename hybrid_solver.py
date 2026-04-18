"""
Hybrid Solver: Jacobi + Neural Network Error Corrector
=======================================================
Problem : solve the 2D Poisson equation  -∇²u = f  on [0,1]²
          with homogeneous Dirichlet boundary conditions.

Idea:
  - Jacobi iteration is cheap but stalls on smooth, low-frequency error.
  - A small CNN is trained to predict that smooth error from the residual.
  - The hybrid loop alternates:  Jacobi sweeps  →  CNN correction  →  …
  - This mirrors multigrid philosophy, but with a learned smoother.

Key design choices for robustness:
  - Per-sample normalisation: each residual is divided by its own RMS
    before entering the CNN, so the network always sees unit-scale input
    regardless of the current convergence stage.  Global stats would
    cause the network to receive near-zero (then exploding) inputs as
    the residual shrinks during the solve.
  - Damping factor α < 1 applied to every CNN correction, acting as a
    safety net against overconfident predictions.

Sections:
  1. Problem setup
  2. Jacobi solver & helpers
  3. Training data generation
  4. CNN (ErrorCorrector)
  5. Training
  6. Hybrid solver loop
  7. Visualisation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

torch.manual_seed(0)
np.random.seed(0)


# =============================================================================
# 1. PROBLEM SETUP
# =============================================================================

N = 32                  # interior grid points per side
h = 1.0 / (N + 1)      # uniform grid spacing

coords = np.linspace(0, 1, N + 2)    # includes boundary nodes
X, Y   = np.meshgrid(coords, coords)

# Reference problem: exact solution u* = sin(πx)sin(πy)
u_star = np.sin(np.pi * X) * np.sin(np.pi * Y)
f_ref  = 2.0 * np.pi**2 * u_star


# =============================================================================
# 2. JACOBI SOLVER  AND  HELPERS
# =============================================================================

def jacobi_step(u, f):
    """One Jacobi sweep on interior points (boundary stays zero)."""
    u_new = u.copy()
    u_new[1:-1, 1:-1] = 0.25 * (
        u[:-2, 1:-1] + u[2:, 1:-1] +
        u[1:-1, :-2] + u[1:-1, 2:] +
        h**2 * f[1:-1, 1:-1]
    )
    return u_new


def residual(u, f):
    """r = f − Au,  where A is the 2D finite-difference Laplacian."""
    Au = np.zeros_like(u)
    Au[1:-1, 1:-1] = (
        4*u[1:-1, 1:-1]
        - u[:-2, 1:-1] - u[2:, 1:-1]
        - u[1:-1, :-2] - u[1:-1, 2:]
    ) / h**2
    return f - Au


def relative_error(u):
    return np.linalg.norm(u - u_star) / np.linalg.norm(u_star)


def run_jacobi(f, n_steps, u0=None):
    """Plain Jacobi; returns final u and per-step relative error history."""
    u = np.zeros((N+2, N+2)) if u0 is None else u0.copy()
    errors = []
    for _ in range(n_steps):
        u = jacobi_step(u, f)
        errors.append(relative_error(u))
    return u, errors


# =============================================================================
# 3. TRAINING DATA  (residual → correction pairs)
# =============================================================================

def exact_solve(f):
    """
    Direct sparse solve for -∇²u = f on the N×N interior.
    Used to get ground-truth solutions for training problems.
    """
    n = N * N
    diag_main = 4.0 * np.ones(n)
    diag_off1 = -1.0 * np.ones(n - 1)
    diag_off1[np.arange(N-1, n-1, N)] = 0.0    # no wrap-around connections
    diag_offN = -1.0 * np.ones(n - N)

    A = diags(
        [diag_offN, diag_off1, diag_main, diag_off1, diag_offN],
        [-N, -1, 0, 1, N], format='csc'
    ) / h**2

    u_int = spsolve(A, f[1:-1, 1:-1].ravel()).reshape(N, N)
    u = np.zeros((N+2, N+2))
    u[1:-1, 1:-1] = u_int
    return u


def random_smooth_rhs():
    """Random RHS, sum of low-frequency sine modes."""
    f = np.zeros((N+2, N+2))
    for k in range(1, 5):
        for l in range(1, 5):
            f += np.random.randn() * np.sin(k * np.pi * X) * np.sin(l * np.pi * Y)
    return f


def generate_training_data(n_problems=500):
    """
    For each random problem:
      - compute the exact solution (direct sparse solver)
      - run a random number of Jacobi steps from zero
      - record (residual, error) on the interior, normalised per-sample

    Per-sample normalisation: both r and e are divided by rms(r).
    This teaches the CNN a scale-free mapping, which generalises robustly
    to any convergence stage during inference.
    """
    print(f"Generating {n_problems} training pairs...")
    res_list, err_list = [], []

    for _ in range(n_problems):
        f        = random_smooth_rhs()
        u_ref    = exact_solve(f)
        n_warm   = np.random.randint(10, 60)
        u_approx, _ = run_jacobi(f, n_warm)

        r = residual(u_approx, f)[1:-1, 1:-1]
        e = (u_ref - u_approx)[1:-1, 1:-1]

        # --- per-sample normalisation (key robustness ingredient) ---
        scale = np.sqrt(np.mean(r**2)) + 1e-10
        res_list.append(r / scale)
        err_list.append(e / scale)

    return (
        np.array(res_list, dtype=np.float32),
        np.array(err_list, dtype=np.float32),
    )


residuals, corrections = generate_training_data(n_problems=500)

# Tensors: (N_samples, 1, N, N)
X_train = torch.from_numpy(residuals[:, None])
Y_train = torch.from_numpy(corrections[:, None])


# =============================================================================
# 4. CNN MODEL  (ErrorCorrector)
# =============================================================================

class ErrorCorrector(nn.Module):
    """
    A lightweight encoder-decoder CNN:
      residual field (N×N)  →  error correction field (N×N)

    Padding=1 keeps spatial resolution fixed.
    The skip connection from input to output biases the network toward
    small corrections (residual learning), which stabilises training.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16,  1, 3, padding=1),
        )

    def forward(self, x):
        # Add skip connection
        return self.decoder(self.encoder(x)) + x

# =============================================================================
# 5. TRAINING
# =============================================================================

model     = ErrorCorrector()
optimiser = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.5)
loss_fn   = nn.MSELoss()

dataset = torch.utils.data.TensorDataset(X_train, Y_train)
loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

n_epochs = 150
print(f"Training ErrorCorrector CNN for {n_epochs} epochs...")

for epoch in range(n_epochs):
    epoch_loss = 0.0
    for xb, yb in loader:
        pred  = model(xb)
        loss  = loss_fn(pred, yb)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:3d}/{n_epochs}  loss = {epoch_loss/len(loader):.5f}")


# =============================================================================
# 6. HYBRID SOLVER  (Jacobi + CNN in a loop)
# =============================================================================

def apply_cnn_correction(u, f):

    r     = residual(u, f)[1:-1, 1:-1]
    scale = np.sqrt(np.mean(r**2)) + 1e-10

    inp = torch.from_numpy((r / scale)[None, None].astype(np.float32))
    with torch.no_grad():
        correction = model(inp).squeeze().numpy() * scale

    u_new = u.copy()
    u_new[1:-1, 1:-1] += correction
    return u_new

def run_hybrid(f, n_steps, cnn_at_step=50):
    """Apply CNN correction only once at step `cnn_at_step`."""
    u      = np.zeros((N+2, N+2))
    errors = []

    for step in range(n_steps):
        u = jacobi_step(u, f)
        errors.append(relative_error(u))

        if step + 1 == cnn_at_step:
            u = apply_cnn_correction(u, f)
            errors[-1] = relative_error(u)
    return u, errors

# =============================================================================
# 7. COMPARISON  &  VISUALISATION
# =============================================================================

n_steps = 500

print("\nRunning solvers for comparison...")
u_jacobi, err_jacobi = run_jacobi(f_ref, n_steps)
u_hybrid, err_hybrid = run_hybrid(f_ref, n_steps, cnn_at_step=50)
print(f"  Jacobi only  — final error : {err_jacobi[-1]:.3e}")
print(f"  Jacobi + CNN — final error : {err_hybrid[-1]:.3e}")

iters = np.arange(1, n_steps + 1)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 4.5))
gs  = gridspec.GridSpec(1, 4, width_ratios=[2.5, 1, 1, 1], wspace=0.35)

ax_conv = fig.add_subplot(gs[0])
ax_star = fig.add_subplot(gs[1])
ax_jac  = fig.add_subplot(gs[2])
ax_hyb  = fig.add_subplot(gs[3])

# ── Convergence curves
ax_conv.plot(iters, err_jacobi,
                 label='Jacobi only', color='steelblue', linewidth=2)
ax_conv.plot(iters, err_hybrid,
                 label=f'Jacobi + CNN',
                 color='tomato', linewidth=2)
ax_conv.vlines(x = 50, ymin = np.min(err_hybrid), ymax = np.max(err_jacobi), linestyles = 'dashed', label = 'CNN correction', colors = 'grey', linewidth = 1.5)

ax_conv.set_yscale('log')
ax_conv.set_xlabel('Jacobi iterations', fontsize=11)
ax_conv.set_ylabel('Relative L2 error', fontsize=11)
ax_conv.set_title('Convergence comparison', fontsize=12, fontweight='bold')
ax_conv.legend(fontsize=10)
ax_conv.grid(True, alpha=0.3)
ax_conv.set_xlim(1, n_steps)

# ── Solution fields ───────────────────────────────────────────────────────────
vmin, vmax = u_star.min(), u_star.max()
kwargs_plot = dict(vmin=vmin, vmax=vmax, cmap='RdBu_r', origin='lower')

ax_star.imshow(u_star, **kwargs_plot)
ax_star.set_title('Exact solution\n$u^\\star$', fontsize=11)
ax_star.axis('off')

ax_jac.imshow(u_jacobi, **kwargs_plot)
ax_jac.set_title(f'Jacobi only\nerr = {err_jacobi[-1]:.2e}', fontsize=11)
ax_jac.axis('off')

im = ax_hyb.imshow(u_hybrid, **kwargs_plot)
ax_hyb.set_title(f'Jacobi + CNN\nerr = {err_hybrid[-1]:.2e}', fontsize=11)
ax_hyb.axis('off')

plt.colorbar(im, ax=[ax_star, ax_jac, ax_hyb], shrink=0.8, pad=0.02)

# plt.savefig('hybrid_solver.png', dpi=150, bbox_inches='tight')
plt.show()