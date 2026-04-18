
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Arrow3D(FancyArrowPatch):
    """Helper class for 3D arrows"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return min(zs)

def plot_training_curves(weak_results, strong_results, lambda_div=0.1):
    """Plot training and validation curves for both models"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Weak constraint model
    epochs = range(1, len(weak_results[0]) + 1)
    
    axes[0, 0].plot(epochs, weak_results[0], 'b-', label='Training Loss', alpha=0.7)
    axes[0, 0].plot(epochs, weak_results[1], 'r-', label='Validation Loss', alpha=0.7)
    axes[0, 0].set_title(f'Weak Constraint (lambda={lambda_div})')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Strong constraint model
    epochs = range(1, len(strong_results[0]) + 1)
    
    axes[0, 1].plot(epochs, strong_results[0], 'b-', label='Training Loss', alpha=0.7)
    axes[0, 1].plot(epochs, strong_results[1], 'r-', label='Validation Loss', alpha=0.7)
    axes[0, 1].set_title('Strong Constraint')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Comparison
    axes[0, 2].plot(weak_results[1], 'b-', label=f'Weak (lambda={lambda_div})', alpha=0.7)
    axes[0, 2].plot(strong_results[1], 'r-', label='Strong', alpha=0.7)
    axes[0, 2].set_title('Validation Loss Comparison')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Validation MSE')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    # Divergence comparison
    axes[1, 0].plot(weak_results[2], 'b-', label=f'Weak (lambda={lambda_div})', alpha=0.7)
    axes[1, 0].set_title('Weak Constraint: Divergence')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean Squared Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].plot(strong_results[2], 'r-', label='Strong', alpha=0.7)
    axes[1, 1].set_title('Strong Constraint: Divergence')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Mean Squared Divergence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # Final divergence comparison
    weak_final_div = weak_results[2][-1]
    strong_final_div = strong_results[2][-1]
    
    models = ['Weak', 'Strong']
    div_values = [weak_final_div, strong_final_div]
    
    axes[1, 2].bar(models, div_values, color=['blue', 'red'], alpha=0.7)
    axes[1, 2].set_title('Final Divergence Comparison')
    axes[1, 2].set_ylabel('Mean Divergence²')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Add values on top of bars
    for i, v in enumerate(div_values):
        axes[1, 2].text(i, v, f'{v:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_velocity_fields(weak_model, strong_model, x_grid, 
                        extent=(-1, 1, -1, 1), n_points=50):
    """Plot velocity fields for both models"""
    # Create grid for plotting
    x = np.linspace(extent[0], extent[1], n_points)
    y = np.linspace(extent[2], extent[3], n_points)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    grid_tensor = torch.FloatTensor(grid_points).to(device)
    
    # Predict velocities
    with torch.no_grad():
        weak_v_pred = weak_model.network(grid_tensor).cpu().numpy()
        strong_v_pred = strong_model.compute_velocity(grid_tensor).cpu().numpy()
    
    # True velocity
    v_x = -6*np.pi*np.sin(4*np.pi*X) * np.sin(6*np.pi*Y)
    v_y = -4*np.pi*np.cos(4*np.pi*X) * np.cos(6*np.pi*Y)
    true_v = np.stack((v_x.ravel(), v_y.ravel()), axis = 1)
    # Reshape for plotting
    weak_U = weak_v_pred[:, 0].reshape(X.shape)
    weak_V = weak_v_pred[:, 1].reshape(X.shape)
    strong_U = strong_v_pred[:, 0].reshape(X.shape)
    strong_V = strong_v_pred[:, 1].reshape(X.shape)
    true_U = true_v[:, 0].reshape(X.shape)
    true_V = true_v[:, 1].reshape(X.shape)
    
    # Compute magnitude
    weak_mag = np.sqrt(weak_U**2 + weak_V**2)
    strong_mag = np.sqrt(strong_U**2 + strong_V**2)
    true_mag = np.sqrt(true_U**2 + true_V**2)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # True velocity
    im1 = axes[0, 0].imshow(true_mag, extent=extent, origin='lower', cmap='viridis')
    axes[0, 0].quiver(X[::4, ::4], Y[::4, ::4], 
                     true_U[::4, ::4], true_V[::4, ::4], 
                     color='white', alpha=0.7)
    axes[0, 0].set_title('True Velocity Field')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Weak model velocity
    im2 = axes[0, 1].imshow(weak_mag, extent=extent, origin='lower', cmap='viridis')
    axes[0, 1].quiver(X[::4, ::4], Y[::4, ::4], 
                     weak_U[::4, ::4], weak_V[::4, ::4], 
                     color='white', alpha=0.7)
    axes[0, 1].set_title('Weak Constraint: Predicted Velocity')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Strong model velocity
    im3 = axes[0, 2].imshow(strong_mag, extent=extent, origin='lower', cmap='viridis')
    axes[0, 2].quiver(X[::4, ::4], Y[::4, ::4], 
                     strong_U[::4, ::4], strong_V[::4, ::4], 
                     color='white', alpha=0.7)
    axes[0, 2].set_title('Strong Constraint: Predicted Velocity')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Error plots
    weak_error = np.sqrt((weak_U - true_U)**2 + (weak_V - true_V)**2)
    strong_error = np.sqrt((strong_U - true_U)**2 + (strong_V - true_V)**2)
    
    im4 = axes[1, 0].imshow(weak_error, extent=extent, origin='lower', cmap='Reds')
    axes[1, 0].set_title('Weak Constraint: Velocity Error')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(strong_error, extent=extent, origin='lower', cmap='Reds')
    axes[1, 1].set_title('Strong Constraint: Velocity Error')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Error difference
    error_diff = strong_error - weak_error
    im6 = axes[1, 2].imshow(error_diff, extent=extent, origin='lower', 
                           cmap='RdBu_r', vmin=-np.max(np.abs(error_diff)), 
                           vmax=np.max(np.abs(error_diff)))
    axes[1, 2].set_title('Error Difference (Strong - Weak)')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im6, ax=axes[1, 2])
    
    # Divergence plots
    # Compute divergence for weak model
    with torch.set_grad_enabled(True):
        grid_tensor_grad = grid_tensor.clone().requires_grad_(True)
        weak_v = weak_model.network(grid_tensor_grad)
        vx = weak_v[:, 0]
        vy = weak_v[:, 1]
        
        grad_vx = torch.autograd.grad(
            vx.sum(), grid_tensor_grad, create_graph=False, retain_graph=True
        )[0]
        grad_vy = torch.autograd.grad(
            vy.sum(), grid_tensor_grad, create_graph=False, retain_graph=False
        )[0]
        
        weak_div = (grad_vx[:, 0] + grad_vy[:, 1]).cpu().numpy().reshape(X.shape)
    
    # Stream function for strong model
    with torch.no_grad():
        grid_tensor_psi = grid_tensor.clone().requires_grad_(True)
        psi = strong_model.network(grid_tensor_psi).cpu().numpy().reshape(X.shape)
    
    im7 = axes[2, 0].imshow(weak_div, extent=extent, origin='lower', 
                           cmap='RdBu_r', vmin=-np.max(np.abs(weak_div)), 
                           vmax=np.max(np.abs(weak_div)))
    axes[2, 0].set_title('Weak Constraint: Divergence')
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('y')
    plt.colorbar(im7, ax=axes[2, 0])
    
    im8 = axes[2, 1].contourf(X, Y, psi, levels=20, cmap='viridis')
    axes[2, 1].set_title('Strong Constraint: Stream Function Psi')
    axes[2, 1].set_xlabel('x')
    axes[2, 1].set_ylabel('y')
    plt.colorbar(im8, ax=axes[2, 1])
    
    # Streamlines
    axes[2, 2].streamplot(X, Y, strong_U, strong_V, color='blue', linewidth=0.5, 
                         density=2, arrowstyle='->', arrowsize=1.5)
    axes[2, 2].contour(X, Y, psi, levels=20, colors='red', linewidths=0.5, alpha=0.7)
    axes[2, 2].set_title('Strong: Streamlines and Psi Contours')
    axes[2, 2].set_xlabel('x')
    axes[2, 2].set_ylabel('y')
    axes[2, 2].set_xlim(extent[0], extent[1])
    axes[2, 2].set_ylim(extent[2], extent[3])
    
    plt.tight_layout()
    plt.show()
    
    return {
        'weak_velocity': (weak_U, weak_V),
        'strong_velocity': (strong_U, strong_V),
        'true_velocity': (true_U, true_V),
        'weak_divergence': weak_div,
        'stream_function': psi,
        'weak_error': weak_error,
        'strong_error': strong_error
    }

def plot_pointwise_comparison(weak_model, strong_model, x_points):
    """Plot pointwise comparisons at specific locations"""
    x_tensor = torch.FloatTensor(x_points).to(device)
    
    with torch.no_grad():
        weak_pred = weak_model.network(x_tensor).cpu().numpy()
        strong_pred = strong_model.compute_velocity(x_tensor).cpu().numpy()
    
    v_x = -6*np.pi*np.sin(4*np.pi*x_points[:,0]) * np.sin(6*np.pi*x_points[:,1])
    v_y = -4*np.pi*np.cos(4*np.pi*x_points[:,0]) * np.cos(6*np.pi*x_points[:,1])
    true_v = np.stack((v_x.ravel(), v_y.ravel()), axis = 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Scatter plots of predictions vs true
    axes[0, 0].scatter(true_v[:, 0], weak_pred[:, 0], alpha=0.5, label='v_x')
    axes[0, 0].scatter(true_v[:, 1], weak_pred[:, 1], alpha=0.5, label='v_y')
    axes[0, 0].plot([true_v.min(), true_v.max()], [true_v.min(), true_v.max()], 
                   'r--', alpha=0.5)
    axes[0, 0].set_title('Weak Constraint: Predicted vs True')
    axes[0, 0].set_xlabel('True Velocity')
    axes[0, 0].set_ylabel('Predicted Velocity')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(true_v[:, 0], strong_pred[:, 0], alpha=0.5, label='v_x')
    axes[0, 1].scatter(true_v[:, 1], strong_pred[:, 1], alpha=0.5, label='v_y')
    axes[0, 1].plot([true_v.min(), true_v.max()], [true_v.min(), true_v.max()], 
                   'r--', alpha=0.5)
    axes[0, 1].set_title('Strong Constraint: Predicted vs True')
    axes[0, 1].set_xlabel('True Velocity')
    axes[0, 1].set_ylabel('Predicted Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    weak_error = np.linalg.norm(weak_pred - true_v, axis=1)
    strong_error = np.linalg.norm(strong_pred - true_v, axis=1)
    
    axes[0, 2].hist(weak_error, bins=30, alpha=0.5, label='Weak', density=True)
    axes[0, 2].hist(strong_error, bins=30, alpha=0.5, label='Strong', density=True)
    axes[0, 2].set_title('Error Distribution')
    axes[0, 2].set_xlabel('Error Magnitude')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Divergence at points
    with torch.set_grad_enabled(True):
        x_tensor_grad = x_tensor.clone().requires_grad_(True)
        weak_v = weak_model.network(x_tensor_grad)
        vx = weak_v[:, 0]
        vy = weak_v[:, 1]
        
        grad_vx = torch.autograd.grad(
            vx.sum(), x_tensor_grad, create_graph=False, retain_graph=True
        )[0]
        grad_vy = torch.autograd.grad(
            vy.sum(), x_tensor_grad, create_graph=False, retain_graph=False
        )[0]
        
        weak_div = (grad_vx[:, 0] + grad_vy[:, 1]).cpu().numpy()
    
    axes[1, 0].scatter(x_points[:, 0], x_points[:, 1], c=weak_div, 
                      cmap='RdBu_r', s=50, alpha=0.8)
    axes[1, 0].set_title('Weak Constraint: Pointwise Divergence')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    
    # Error magnitude at points
    axes[1, 1].scatter(x_points[:, 0], x_points[:, 1], c=weak_error, 
                      cmap='Reds', s=50, alpha=0.8, label='Weak')
    axes[1, 1].set_title('Weak Constraint: Pointwise Error')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    
    axes[1, 2].scatter(x_points[:, 0], x_points[:, 1], c=strong_error, 
                      cmap='Reds', s=50, alpha=0.8, label='Strong')
    axes[1, 2].set_title('Strong Constraint: Pointwise Error')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("="*60)
    print("MODEL COMPARISON STATISTICS")
    print("="*60)
    print(f"Weak Constraint Model:")
    print(f"  MSE: {np.mean((weak_pred - true_v)**2):.6f}")
    print(f"  Std Error: {np.std(weak_error):.6f}")
    print(f"  Max Error: {np.max(weak_error):.6f}")
    print(f"  Mean |Divergence|: {np.mean(np.abs(weak_div)):.6e}")
    print()
    print(f"Strong Constraint Model:")
    print(f"  MSE: {np.mean((strong_pred - true_v)**2):.6f}")
    print(f"  Std Error: {np.std(strong_error):.6f}")
    print(f"  Max Error: {np.max(strong_error):.6f}")

def plot_multi_accuracy_vs_points(nb_points, results_val, results_div, results_train, legends, colors):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for i in range(len(results_val)):
        ax.plot(nb_points, results_val[i], label="Validation Loss - " + legends[i], color=colors[i])
        ax.plot(nb_points, results_train[i], label="Train Loss - " + legends[i], linestyle="--", color=colors[i])
    ax.set_yscale('log')
    ax.set_ylabel('Error')
    ax.set_xlabel("Number of sampled points")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True)
    ax = axes[1]
    for i in range(len(results_div)):
        ax.plot(nb_points, results_div[i], label="Divergence - " + legends[i], color=colors[i])
    ax.set_yscale('log')
    ax.set_ylabel("Divergence norm")
    ax.set_xlabel("Number of sampled points")
    ax.set_title("Divergence")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()
