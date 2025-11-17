"""
This script implements the PageRank algorithm using eigenvalue decomposition
and iterative methods to rank web pages in a network.
"""

import numpy as np
from scipy.linalg import eig

def main():
    """
    Implements PageRank Algorithm 
    
    The PageRank algorithm ranks websites based on the number of links pointing
    to them. Uses a stochastic matrix where each entry M[i,j] is the probability
    that a user on page j will click a link to page i.
    """
    print("=" * 60)
    print("PART 1: PAGERANK ALGORITHM")
    print("=" * 60)
    
    # Define the stochastic matrix M as given in the assignment
    M = np.array([
        [0, 0, 1/2, 0],
        [1/3, 0, 0, 1/2],
        [1/3, 1/2, 0, 1/2],
        [1/3, 1/2, 1/2, 0]
    ])
    
    print("Stochastic Matrix M:")
    print(M)
    print()
    
    # 1. Compute dominant eigenvector using scipy.linalg.eig
    print("1. Computing dominant eigenvector using scipy.linalg.eig:")
    eigenvalues, eigenvectors = eig(M)
    
    # Find the dominant eigenvalue (should be close to 1.0)
    dominant_idx = np.argmax(np.real(eigenvalues))
    dominant_eigenvalue = eigenvalues[dominant_idx]
    dominant_eigenvector = np.real(eigenvectors[:, dominant_idx])
    
    print(f"   Dominant eigenvalue: {dominant_eigenvalue:.6f}")
    print(f"   Raw dominant eigenvector: {dominant_eigenvector}")
    
    # 2. Iterative method starting with initial rank vector of ones
    print("\n2. Iterative method starting with initial rank vector:")
    v = np.ones(4)  # Initial rank vector
    tolerance = 1e-10
    max_iterations = 1000
    
    print(f"   Initial vector: {v}")
    
    for i in range(max_iterations):
        v_new = M @ v
        # Check for convergence (difference between consecutive vectors is small)
        if np.linalg.norm(v_new - v) < tolerance:
            print(f"   Converged after {i+1} iterations")
            break
        v = v_new
    
    print(f"   Final vector before normalization: {v}")
    
    # 3. Normalize so sum equals 1
    print("\n3. Normalizing eigenvectors so sum of all ranks is 1:")
    pagerank_scores_eig = np.abs(dominant_eigenvector) / np.sum(np.abs(dominant_eigenvector))
    pagerank_scores_iter = v / np.sum(v)
    
    print(f"   PageRank scores (eigenvalue method): {pagerank_scores_eig}")
    print(f"   PageRank scores (iterative method): {pagerank_scores_iter}")
    print(f"   Sum of scores: {np.sum(pagerank_scores_eig):.6f}")
    
   
    # 4. Analysis of results - which page is ranked highest and why
    highest_ranked_page = np.argmax(pagerank_scores_eig)
    
    # Check if there's a tie
    max_score = np.max(pagerank_scores_eig)
    tied_pages = np.where(np.isclose(pagerank_scores_eig, max_score))[0]
    
    if len(tied_pages) > 1:
        print(f"   Tied for highest rank: Pages {tied_pages} (0-indexed)")
        print(f"   PageRank score: {max_score:.6f}")
    else:
        print(f"   Highest ranked page: Page {highest_ranked_page} (0-indexed)")
        print(f"   PageRank score: {pagerank_scores_eig[highest_ranked_page]:.6f}")
    
    # Display detailed link analysis
    print("\n   Link Analysis:")
    for i in range(4):
        incoming_links = []
        for j in range(4):
            if M[i, j] > 0:
                incoming_links.append(f"Page {j} ({M[i,j]:.3f})")
        print(f"   Page {i} receives links from: {', '.join(incoming_links) if incoming_links else 'None'}")
    
    return pagerank_scores_eig

if __name__ == "__main__":
    scores = main()
    print(f"\nFinal PageRank scores: {scores}")