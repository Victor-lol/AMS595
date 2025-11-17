# AMS 595/DCS 525 Python Project 3: Machine Learning

## Project Overview

This project implements four fundamental machine learning algorithms from scratch using Python and scientific computing libraries:

1. **PageRank Algorithm**  - Web page ranking using eigenvalue decomposition
2. **Principal Component Analysis (PCA)**  - Dimensionality reduction techniques
3. **Linear Regression via Least Squares**  - House price prediction model
4. **Gradient Descent Optimization**  - Loss function minimization

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Dependencies

Install the required packages using pip:

```bash
pip install numpy scipy matplotlib
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

### Alternative: Using Conda

If you prefer using Conda:

```bash
conda install numpy scipy matplotlib
```

## How to Run

Each part is implemented as a standalone script. Run them individually:

### Part 1: PageRank Algorithm

```bash
python part1_pagerank.py
```

**What it does:**
- Computes PageRank scores using eigenvalue decomposition
- Implements iterative power method for comparison
- Analyzes web page ranking and link structure
- Identifies highest-ranked pages with explanations

**Expected Output:**
- Stochastic matrix representation
- Dominant eigenvalue and eigenvector
- Convergence analysis of iterative method
- Final PageRank scores and ranking analysis

### Part 2: PCA Dimensionality Reduction

```bash
python part2_pca.py
```

**What it does:**
- Generates synthetic height-weight data (100 samples)
- Performs eigenvalue decomposition on covariance matrix
- Reduces 2D data to 1D using principal components
- Creates visualizations showing original vs projected data

**Expected Output:**
- Covariance matrix computation
- Eigenvalues and eigenvectors
- Variance explanation ratios
- Visualization saved as `pca_results.png`

### Part 3: Linear Regression via Least Squares

```bash
python part3_linear_regression.py
```

**What it does:**
- Solves house price prediction problem
- Uses both `scipy.linalg.lstsq` and normal equations
- Predicts price for new house (2400 sqft, 3 bedrooms, 20 years)
- Compares numerical methods and analyzes model performance

**Expected Output:**
- Model coefficients for square footage, bedrooms, and age
- Predicted house price: $1448.75k
- Training set performance metrics
- Method comparison and discussion

### Part 4: Gradient Descent Optimization

```bash
python part4_gradient_descent.py
```

**What it does:**
- Minimizes MSE loss function for 100×50 matrices
- Implements both SciPy optimization and manual gradient descent
- Tracks convergence and creates comprehensive visualizations
- Compares optimization methods

**Expected Output:**
- Optimization progress and convergence analysis
- Final loss values (SciPy: ~1e-28, Manual: ~1e-5)
- Comprehensive visualization saved as `gradient_descent_results.png`
- Performance comparison between methods