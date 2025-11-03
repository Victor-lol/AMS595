# AMS 595 Python Project 2: Fractal Approximation

This repository contains the implementation of three computational mathematics problems: Mandelbrot set visualization, Markov chain analysis, and Taylor series approximation.

## Project Structure

```
595HW4/
├── mandelbrot.py          # Mandelbrot set computation and visualization
├── markov_chain.py        # Markov chain analysis with stationary distribution
├── taylor.py              # Taylor series approximation and analysis
├── report.md              # Project report
└── README.md              # This file
```

## Dependencies

This project requires Python 3.7+ and the following packages:

### Required Packages
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `sympy` - Symbolic mathematics
- `pandas` - Data analysis and CSV handling

### Installation

#### Option 1: Using pip
```bash
pip install numpy matplotlib sympy pandas
```

#### Option 2: Using conda
```bash
conda install numpy matplotlib sympy pandas
```

#### Option 3: Installing all at once
```bash
pip install numpy matplotlib sympy pandas
```

### Requirements File
Alternatively, you can create a `requirements.txt` file:
```
numpy>=1.20.0
matplotlib>=3.3.0
sympy>=1.8.0
pandas>=1.3.0
```


## How to Run

### 1. Mandelbrot Set 
Generates the Mandelbrot fractal visualization:

```bash
python mandelbrot.py
```

**Output:**
- `mandelbrot.png` - High-resolution fractal image
- Interactive plot display

**Expected runtime:** 10-30 seconds depending on system

### 2. Markov Chain Analysis 

Performs complete Markov chain convergence analysis:

```bash
python markov_chain.py
```

**Output:**
- Console output showing:
  - Random transition matrix P
  - Initial probability vector p₀
  - Final probability vector p₅₀ after 50 iterations
  - Stationary distribution
  - Convergence verification within 10⁻⁵ tolerance

**Expected runtime:** < 1 second

### 3. Taylor Series Approximation 

Computes Taylor series approximation with performance analysis:

```bash
python taylor.py
```

**Output:**
- `taylor_approximation.png` - Function vs. approximation plot
- `taylor_values.csv` - Performance analysis data
- Interactive plot display

**Expected runtime:** 30-60 seconds for full analysis

## Output Files Description

### Generated Images
- **mandelbrot.png**: Grayscale visualization of the Mandelbrot set showing fractal boundary structure
- **taylor_approximation.png**: Comparison plot of actual function f(x) = x sin²(x) + cos(x) vs. Taylor approximation

### Generated Data
- **taylor_values.csv**: Contains performance metrics with columns:
  - `degree`: Taylor series degree (50-100, step 10)
  - `error`: Sum of absolute differences between true and approximated values
  - `computation_time`: Time in seconds to compute approximation
