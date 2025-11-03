import numpy as np

def create_random_transition_matrix(n):
    """
    Create a random n x n transition matrix with normalized rows.
    
    Parameters:
    n: number of states
    
    Returns:
    P: normalized transition matrix
    """
    # Generate random matrix
    P = np.random.rand(n, n)
    
    # Normalize each row so that sum equals 1
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / row_sums
    
    return P

def create_random_probability_vector(n):
    """
    Create a random normalized probability vector.
    
    Parameters:
    n: size of vector
    
    Returns:
    p: normalized probability vector
    """
    # Generate random vector
    p = np.random.rand(n)
    
    # Normalize so sum equals 1
    p = p / p.sum()
    
    return p

def apply_transition_rule(P, p, n_iterations):
    """
    Apply the transition rule p_new = P^T * p for n_iterations.
    
    Parameters:
    P: transition matrix
    p: initial probability vector
    n_iterations: number of iterations to apply
    
    Returns:
    p_final: probability vector after n_iterations
    """
    p_current = p.copy()
    
    for i in range(n_iterations):
        p_current = np.dot(P.T, p_current)
    
    return p_current

def compute_stationary_distribution(P):
    """
    Compute the stationary distribution of transition matrix P.
    The stationary distribution is the eigenvector of P^T corresponding to eigenvalue 1.
    
    Parameters:
    P: transition matrix
    
    Returns:
    stationary_dist: normalized stationary distribution
    """
    # Compute eigenvalues and eigenvectors of P^T
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    
    # Find the eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    
    # Get the corresponding eigenvector
    stationary_vector = np.real(eigenvectors[:, idx])
    
    # Normalize the eigenvector so sum equals 1
    stationary_dist = stationary_vector / stationary_vector.sum()
    
    return stationary_dist

def main():
    """
    Main function to execute the Markov chain analysis.
    """
    print("Markov Chain Analysis")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Create random 5x5 transition matrix
    print("1. Creating random 5x5 transition matrix...")
    P = create_random_transition_matrix(5)
    print("Transition matrix P:")
    print(P)
    print(f"Row sums (should all be 1): {P.sum(axis=1)}")
    print()
    
    # 2. Create random probability vector and apply transition rule 50 times
    print("2. Creating random probability vector and applying transition rule...")
    p0 = create_random_probability_vector(5)
    print(f"Initial probability vector p0: {p0}")
    print(f"Sum of p0 (should be 1): {p0.sum()}")
    
    p50 = apply_transition_rule(P, p0, 50)
    print(f"Probability vector after 50 iterations p50: {p50}")
    print(f"Sum of p50 (should be 1): {p50.sum()}")
    print()
    
    # 3. Compute stationary distribution
    print("3. Computing stationary distribution...")
    stationary_dist = compute_stationary_distribution(P)
    print(f"Stationary distribution: {stationary_dist}")
    print(f"Sum of stationary distribution (should be 1): {stationary_dist.sum()}")
    print()
    
    # 4. Compare p50 with stationary distribution
    print("4. Comparing p50 with stationary distribution...")
    difference = np.abs(p50 - stationary_dist)
    print(f"Component-wise absolute differences: {difference}")
    print(f"Maximum difference: {np.max(difference)}")
    
    tolerance = 1e-5
    matches_within_tolerance = np.all(difference < tolerance)
    print(f"Do they match within {tolerance}? {matches_within_tolerance}")
    
    if matches_within_tolerance:
        print("SUCCESS: p50 and stationary distribution match within tolerance!")
    else:
        print("NOTE: p50 and stationary distribution do not match within tolerance.")
        print("This could be due to the random nature of the matrix or insufficient iterations.")

if __name__ == "__main__":
    main()