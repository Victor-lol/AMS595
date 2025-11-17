"""
This script implements Principal Component Analysis (PCA) for dimensionality
reduction on height-weight data, demonstrating eigenvalue decomposition and
data projection onto principal components.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Set random seed for reproducibility
np.random.seed(42)

def main():
    """
    Implements Dimensionality Reduction via PCA
    
    Principal Component Analysis (PCA) is widely used in data analysis for 
    dimensionality reduction, often to compress data while preserving the 
    most important patterns.
    """
    print("=" * 60)
    print("PART 2: DIMENSIONALITY REDUCTION VIA PCA")
    print("=" * 60)
    
    # Generate synthetic height-weight data (already standardized)
    # Creating correlated data to make PCA meaningful
    n_samples = 100
    mean = [0, 0]  # Already standardized, so mean is 0
    cov = [[1, 0.8], [0.8, 1]]  # Strong correlation between height and weight
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    print(f"Generated standardized data for {n_samples} people")
    print(f"Data shape: {data.shape}")
    print(f"Data mean: [{data.mean(axis=0)[0]:.4f}, {data.mean(axis=0)[1]:.4f}]")
    print(f"Data std: [{data.std(axis=0)[0]:.4f}, {data.std(axis=0)[1]:.4f}]")
    
    # 1. Compute the covariance matrix of the data
    print("\n1. Computing covariance matrix:")
    cov_matrix = np.cov(data.T)
    print(f"   Covariance matrix:")
    print(f"   [[{cov_matrix[0,0]:.6f}, {cov_matrix[0,1]:.6f}],")
    print(f"    [{cov_matrix[1,0]:.6f}, {cov_matrix[1,1]:.6f}]]")
    
    # 2. Perform eigenvalue decomposition using scipy.linalg.eigh
    print("\n2. Performing eigenvalue decomposition using scipy.linalg.eigh:")
    eigenvalues, eigenvectors = eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"   Eigenvalues (sorted): [{eigenvalues[0]:.6f}, {eigenvalues[1]:.6f}]")
    print(f"   Eigenvectors:")
    print(f"   First PC:  [{eigenvectors[0,0]:.6f}, {eigenvectors[1,0]:.6f}]")
    print(f"   Second PC: [{eigenvectors[0,1]:.6f}, {eigenvectors[1,1]:.6f}]")
    
    # 3. Identify principal components and explain variance relationship
    print("\n3. Principal components analysis:")
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    
    print(f"   Total variance: {total_variance:.6f}")
    print(f"   Explained variance ratio:")
    print(f"   - First PC: {explained_variance_ratio[0]:.4f} ({explained_variance_ratio[0]*100:.2f}%)")
    print(f"   - Second PC: {explained_variance_ratio[1]:.4f} ({explained_variance_ratio[1]*100:.2f}%)")
    
    # 4. Reduce dataset to 1D by projecting onto first principal component
    print("\n4. Reducing dataset to 1D:")
    first_pc = eigenvectors[:, 0]
    data_1d = data @ first_pc  # Project data onto first principal component
    
    print(f"   First principal component direction: [{first_pc[0]:.4f}, {first_pc[1]:.4f}]")
    print(f"   1D projection shape: {data_1d.shape}")
    print(f"   1D data range: [{data_1d.min():.3f}, {data_1d.max():.3f}]")
    print(f"   1D data std: {data_1d.std():.4f}")
    
    # Plot original data and 1D projection
    print("\n   Creating visualization...")
    plt.figure(figsize=(15, 5))
    
    # Original 2D data
    plt.subplot(1, 3, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.7, c='blue', s=30)
    plt.xlabel('Height (standardized)')
    plt.ylabel('Weight (standardized)')
    plt.title('Original 2D Data\n(Height vs Weight)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add principal component arrows
    origin = np.mean(data, axis=0)
    scale = 2.0
    plt.arrow(origin[0], origin[1], eigenvectors[0,0]*scale, eigenvectors[1,0]*scale, 
              head_width=0.15, head_length=0.15, fc='red', ec='red', linewidth=2, label='PC1')
    plt.arrow(origin[0], origin[1], eigenvectors[0,1]*scale, eigenvectors[1,1]*scale, 
              head_width=0.15, head_length=0.15, fc='orange', ec='orange', linewidth=2, label='PC2')
    plt.legend()
    
    # 1D projection visualization
    plt.subplot(1, 3, 2)
    plt.scatter(data_1d, np.zeros_like(data_1d), alpha=0.7, c='green', s=30)
    plt.xlabel('First Principal Component')
    plt.ylabel('')
    plt.title(f'1D Projection\n({explained_variance_ratio[0]*100:.1f}% of variance)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 0.5)
    
    # Show histogram of 1D projection
    plt.subplot(1, 3, 3)
    plt.hist(data_1d, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('First Principal Component')
    plt.ylabel('Frequency')
    plt.title('Distribution of 1D Projection')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '/Users/victor/Documents/GitHub/AMS595/595HW5/pca_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Plot saved as: {output_file}")
    plt.show()
    
    # Additional analysis: reconstruction error
    print("\n   Additional Analysis - Information Loss:")
    # Reconstruct 2D data from 1D projection
    data_reconstructed = np.outer(data_1d, first_pc)
    reconstruction_error = np.mean((data - data_reconstructed)**2)
    
    print(f"   Reconstruction error (MSE): {reconstruction_error:.6f}")
    print(f"   This represents the information lost by reducing to 1D")
    
    return data, data_1d, eigenvalues, eigenvectors

if __name__ == "__main__":
    original_data, reduced_data, eigenvals, eigenvecs = main()
    print(f"\nPCA Summary:")
    print(f"- Original data: {original_data.shape}")
    print(f"- Reduced data: {reduced_data.shape}")
    print(f"- Variance explained: {eigenvals[0]/(eigenvals[0]+eigenvals[1])*100:.1f}%")