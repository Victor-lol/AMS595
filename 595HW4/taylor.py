import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
import time
from math import factorial

def taylor_approximation(func, start, end, degree, fixed_c, num_points=100):
    """
    Approximate an analytic function using Taylor series.
    
    Parameters:
    func: sympy function to approximate
    start: beginning of interval
    end: end of interval
    degree: degree of Taylor series (m in equation 3.2)
    fixed_c: point to expand around
    num_points: number of points in the interval
    
    Returns:
    x_vals: array of x values
    approximation: array of approximated function values
    """
    # Create symbolic variable
    x = sp.Symbol('x')
    
    # Create array of x values
    x_vals = np.linspace(start, end, num_points)
    
    # Compute derivatives at the expansion point
    derivatives = []
    f_temp = func
    for n in range(degree + 1):
        # Evaluate nth derivative at fixed_c
        deriv_value = float(f_temp.subs(x, fixed_c))
        derivatives.append(deriv_value)
        
        # Compute next derivative
        f_temp = sp.diff(f_temp, x)
    
    # Compute Taylor series approximation for each x value
    approximation = np.zeros(len(x_vals))
    
    for i, x_val in enumerate(x_vals):
        taylor_sum = 0
        for n in range(degree + 1):
            term = derivatives[n] * (x_val - fixed_c)**n / factorial(n)
            taylor_sum += term
        approximation[i] = taylor_sum
    
    return x_vals, approximation

def plot_function_and_approximation():
    """
    Plot the function f(x) = x*sin^2(x) + cos(x) and its Taylor approximation.
    """
    # Define the symbolic function
    x = sp.Symbol('x')
    func = x * sp.sin(x)**2 + sp.cos(x)
    
    # Parameters from the assignment
    start, end = -10, 10
    degree = 99  # 100 terms means degree 99
    fixed_c = 0
    num_points = 100
    
    print("Computing Taylor series approximation...")
    
    # Compute Taylor approximation
    x_vals, taylor_approx = taylor_approximation(func, start, end, degree, fixed_c, num_points)
    
    # Compute true function values
    func_lambdified = sp.lambdify(x, func, 'numpy')
    true_vals = func_lambdified(x_vals)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, true_vals, 'b-', label='Actual f(x) = x*sin²(x) + cos(x)', linewidth=2)
    plt.plot(x_vals, taylor_approx, 'r--', label='Taylor Approximation (100 terms)', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Taylor Series Approximation of f(x) = x*sin²(x) + cos(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(start, end)
    
    # Save the plot
    plt.savefig('taylor_approximation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Taylor series plot saved as 'taylor_approximation.png'")

def factorial_analysis(func, start, end, fixed_c, initial_degree, final_degree, degree_step, num_points=100):
    """
    Analyze Taylor series approximation for various degrees and measure error and computation time.
    
    Parameters:
    func: sympy function to approximate
    start, end: interval bounds
    fixed_c: expansion point
    initial_degree, final_degree, degree_step: degree range and step
    num_points: number of points in domain
    
    Returns:
    df: pandas dataframe with results
    """
    # Create symbolic variable
    x = sp.Symbol('x')
    
    # Create array of x values
    x_vals = np.linspace(start, end, num_points)
    
    # Compute true function values
    func_lambdified = sp.lambdify(x, func, 'numpy')
    true_vals = func_lambdified(x_vals)
    
    # Initialize results lists
    degrees = []
    errors = []
    times = []
    
    # Iterate through different degrees
    for degree in range(initial_degree, final_degree + 1, degree_step):
        print(f"Computing for degree {degree}...")
        
        # Time the computation
        start_time = time.time()
        
        # Compute Taylor approximation
        _, taylor_approx = taylor_approximation(func, start, end, degree, fixed_c, num_points)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Compute sum of absolute differences
        error = np.sum(np.abs(true_vals - taylor_approx))
        
        # Store results
        degrees.append(degree)
        errors.append(error)
        times.append(computation_time)
        
        print(f"  Error: {error:.6f}, Time: {computation_time:.6f} seconds")
    
    # Create pandas dataframe
    df = pd.DataFrame({
        'degree': degrees,
        'error': errors,
        'computation_time': times
    })
    
    return df

def main():
    """
    Main function to execute Taylor series analysis.
    """
    print("Taylor Series Approximation Analysis")
    print("=" * 50)
    
    # Define the symbolic function
    x = sp.Symbol('x')
    func = x * sp.sin(x)**2 + sp.cos(x)
    
    # 1. Plot function and approximation
    print("1. Plotting function and Taylor approximation...")
    plot_function_and_approximation()
    print()
    
    # 2. Factorial analysis
    print("2. Performing factorial analysis...")
    df = factorial_analysis(
        func=func,
        start=-10,
        end=10,
        fixed_c=0,
        initial_degree=50,
        final_degree=100,
        degree_step=10,
        num_points=100
    )
    
    # Save results to CSV
    df.to_csv('taylor_values.csv', index=False)
    print("Results saved to 'taylor_values.csv'")
    print()
    
    # Display results
    print("Analysis Results:")
    print(df)

if __name__ == "__main__":
    main()