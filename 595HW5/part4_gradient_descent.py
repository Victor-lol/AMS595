"""
This script implements gradient descent to minimize a mean squared error (MSE) 
loss function for a 100×50 matrix, demonstrating both scipy.optimize.minimize
and manual gradient descent implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def main():
    """
    Implements Gradient Descent for Minimizing Loss Function
    
    Given a matrix X ∈ R^(100×50), implement gradient descent to minimize:
    f(X) = (1/2) * Σ(X_ij - A_ij)^2
    
    where A is a known matrix of the same size as X.
    """
    print("=" * 60)
    print("PART 4: GRADIENT DESCENT FOR MINIMIZING LOSS FUNCTION")
    print("=" * 60)
    
    # 5. Initialize matrices X and A with random values
    print("Initializing matrices:")
    rows, cols = 100, 50
    A = np.random.randn(rows, cols)  # Target matrix
    X_init = np.random.randn(rows, cols)  # Initial guess for X
    
    print(f"   Matrix dimensions: {rows} × {cols}")
    print(f"   Target matrix A:")
    print(f"   - Mean: {A.mean():.4f}")
    print(f"   - Std:  {A.std():.4f}")
    print(f"   - Min:  {A.min():.4f}")
    print(f"   - Max:  {A.max():.4f}")
    
    print(f"   Initial matrix X:")
    print(f"   - Mean: {X_init.mean():.4f}")
    print(f"   - Std:  {X_init.std():.4f}")
    print(f"   - Min:  {X_init.min():.4f}")
    print(f"   - Max:  {X_init.max():.4f}")
    
    # 1. Define the loss function: f(X) = (1/2) * Σ(X_ij - A_ij)^2
    def loss_function(X_flat):
        """
        Compute MSE loss between X and target matrix A
        Args:
            X_flat: Flattened version of matrix X
        Returns:
            Loss value (scalar)
        """
        X = X_flat.reshape(rows, cols)
        return 0.5 * np.sum((X - A)**2)
    
    # 2. The gradient of the loss function: ∇f(X) = X - A
    def gradient_function(X_flat):
        """
        Compute gradient of loss function
        Args:
            X_flat: Flattened version of matrix X
        Returns:
            Gradient as flattened array
        """
        X = X_flat.reshape(rows, cols)
        grad = X - A
        return grad.flatten()
    
    # Calculate initial loss
    initial_loss = loss_function(X_init.flatten())
    print(f"\nInitial loss: f(X_0) = {initial_loss:.6f}")
    
    # Theoretical minimum (when X = A)
    theoretical_min = 0.0
    print(f"Theoretical minimum: f(A) = {theoretical_min:.6f}")
    
    # 3. Implement gradient descent using scipy.optimize.minimize
    print(f"\n3. Implementing gradient descent using scipy.optimize.minimize:")
    
    # Track optimization progress
    loss_history = []
    
    def callback_function(X_flat):
        """Callback to track optimization progress"""
        loss = loss_function(X_flat)
        loss_history.append(loss)
        if len(loss_history) % 50 == 0:
            print(f"   Iteration {len(loss_history):3d}: Loss = {loss:.10f}")
    
    print("   Starting optimization...")
    
    result = minimize(
        fun=loss_function,
        x0=X_init.flatten(),
        jac=gradient_function,
        method='BFGS',  # Quasi-Newton method
        callback=callback_function,
        options={
            'maxiter': 1000,
            'gtol': 1e-6,      # Gradient tolerance
            'ftol': 1e-6       # Function tolerance
        }
    )
    
    # 4. Check stopping criteria
    print(f"\n4. Optimization results:")
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")
    print(f"   Number of iterations: {result.nit}")
    print(f"   Number of function evaluations: {result.nfev}")
    print(f"   Number of gradient evaluations: {result.njev}")
    print(f"   Final loss: {result.fun:.2e}")
    print(f"   Difference from theoretical minimum: {abs(result.fun - theoretical_min):.2e}")
    
    # Check convergence criteria
    loss_diff = abs(loss_history[-2] - loss_history[-1]) if len(loss_history) >= 2 else float('inf')
    print(f"   Final loss difference between consecutive iterations: {loss_diff:.2e}")
    print(f"   Threshold (10^-6): {1e-6:.0e}")
    
    if loss_diff < 1e-6:
        print(f"   ✓ Converged: Loss difference < threshold")
    elif result.nit >= 1000:
        print(f"   ⚠ Stopped: Maximum iterations (1000) reached")
    else:
        print(f"   ✓ Converged: Gradient tolerance satisfied")
    
    # Reshape optimized solution back to matrix form
    X_optimized = result.x.reshape(rows, cols)
    
    # Verify that optimized X is close to A
    print(f"\nVerification (X_optimized should be close to A):")
    difference = np.abs(X_optimized - A)
    print(f"   Max absolute difference |X_opt - A|: {np.max(difference):.2e}")
    print(f"   Mean absolute difference:            {np.mean(difference):.2e}")
    print(f"   RMS difference:                      {np.sqrt(np.mean(difference**2)):.2e}")
    
    if np.max(difference) < 1e-6:
        print(f"   ✓ Excellent: X_optimized is very close to A")
    elif np.max(difference) < 1e-3:
        print(f"   ✓ Good: X_optimized is reasonably close to A")
    else:
        print(f"   ⚠ Warning: X_optimized may not be close enough to A")
    
    # 6. Visualize results
    print(f"\n6. Creating visualizations...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Loss convergence (linear scale)
    plt.subplot(2, 3, 1)
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Convergence (Linear Scale)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss convergence (log scale)
    plt.subplot(2, 3, 2)
    plt.semilogy(loss_history, 'r-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Convergence (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Comparison of matrices A vs X_optimized (sample)
    sample_indices = np.random.choice(rows*cols, size=min(1000, rows*cols), replace=False)
    A_sample = A.flatten()[sample_indices]
    X_opt_sample = X_optimized.flatten()[sample_indices]
    
    plt.subplot(2, 3, 3)
    plt.scatter(A_sample, X_opt_sample, alpha=0.6, s=10)
    plt.plot([A_sample.min(), A_sample.max()], [A_sample.min(), A_sample.max()], 'r--', linewidth=2)
    plt.xlabel('A (target values)')
    plt.ylabel('X_optimized (fitted values)')
    plt.title('Target vs Optimized Values')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of differences
    plt.subplot(2, 3, 4)
    plt.hist(difference.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('|X_optimized - A|')
    plt.ylabel('Frequency')
    plt.title('Distribution of Absolute Differences')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Gradient norm during optimization
    gradients_norm = []
    X_current = X_init.copy()
    for i in range(min(len(loss_history), 100)):  # Sample gradients
        grad = gradient_function(X_current.flatten())
        gradients_norm.append(np.linalg.norm(grad))
        # Simulate one step of gradient descent
        if i < len(loss_history) - 1:
            X_current = X_current - 0.01 * grad.reshape(rows, cols)
    
    plt.subplot(2, 3, 5)
    plt.semilogy(gradients_norm[:len(gradients_norm)], 'g-', linewidth=2)
    plt.xlabel('Iteration (sampled)')
    plt.ylabel('||Gradient|| (log scale)')
    plt.title('Gradient Norm During Optimization')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Final optimization summary
    plt.subplot(2, 3, 6)
    stats_text = f"""Optimization Summary:
    
Initial Loss: {initial_loss:.2e}
Final Loss: {result.fun:.2e}
Iterations: {result.nit}
Function Evals: {result.nfev}
Gradient Evals: {result.njev}

Final Statistics:
Max |X-A|: {np.max(difference):.2e}
Mean |X-A|: {np.mean(difference):.2e}
RMS |X-A|: {np.sqrt(np.mean(difference**2)):.2e}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Optimization Summary')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '/Users/victor/Documents/GitHub/AMS595/595HW5/gradient_descent_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved as: {output_file}")
    plt.show()
    
    # Bonus: Manual gradient descent implementation for comparison
    print(f"\n" + "="*50)
    print("BONUS: MANUAL GRADIENT DESCENT IMPLEMENTATION")
    print("="*50)
    
    X_manual = X_init.copy()
    learning_rate = 0.01
    tolerance = 1e-6
    max_iterations = 1000
    loss_history_manual = []
    
    print(f"Manual gradient descent parameters:")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Tolerance: {tolerance}")
    print(f"   Max iterations: {max_iterations}")
    
    print(f"\nRunning manual gradient descent...")
    for i in range(max_iterations):
        # Compute loss and gradient
        loss = 0.5 * np.sum((X_manual - A)**2)
        gradient = X_manual - A
        
        loss_history_manual.append(loss)
        
        # Check for convergence
        if i > 0 and abs(loss_history_manual[-2] - loss_history_manual[-1]) < tolerance:
            print(f"   Manual GD converged after {i+1} iterations")
            break
        
        # Update X using gradient descent
        X_manual = X_manual - learning_rate * gradient
        
        if (i+1) % 200 == 0:
            print(f"   Iteration {i+1:4d}: Loss = {loss:.6f}")
    
    final_loss_manual = 0.5 * np.sum((X_manual - A)**2)
    manual_difference = np.abs(X_manual - A)
    
    print(f"\nManual gradient descent results:")
    print(f"   Final loss: {final_loss_manual:.2e}")
    print(f"   Max |X-A|: {np.max(manual_difference):.2e}")
    print(f"   Mean |X-A|: {np.mean(manual_difference):.2e}")
    
    print(f"\nComparison: SciPy vs Manual Implementation:")
    print(f"   SciPy final loss:  {result.fun:.2e}")
    print(f"   Manual final loss: {final_loss_manual:.2e}")
    print(f"   SciPy iterations:  {result.nit}")
    print(f"   Manual iterations: {len(loss_history_manual)}")
    print(f"   ")
    print(f"   SciPy uses sophisticated optimization (BFGS) with adaptive step sizes,")
    print(f"   while manual implementation uses fixed learning rate gradient descent.")
    
    return result, loss_history, X_optimized, A

if __name__ == "__main__":
    optimization_result, loss_hist, X_opt, A_target = main()
    print(f"\nFinal Summary:")
    print(f"- Optimization successful: {optimization_result.success}")
    print(f"- Final loss: {optimization_result.fun:.2e}")
    print(f"- Converged in {optimization_result.nit} iterations")