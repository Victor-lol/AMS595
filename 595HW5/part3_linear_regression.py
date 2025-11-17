"""
This script implements linear regression using least squares to predict house
prices based on features like square footage, number of bedrooms, and age.
"""

import numpy as np
from scipy.linalg import lstsq, solve

def main():
    """
    Implements Linear Regression via Least Squares (30 points)
    
    A real estate company wants to predict house prices based on features like 
    square footage, number of bedrooms, and age of the house.
    """
    print("=" * 60)
    print("PART 3: LINEAR REGRESSION VIA LEAST SQUARES")
    print("=" * 60)
    
    # Given data from assignment
    X = np.array([
        [2100, 3, 20],  # sqft, bedrooms, age
        [2500, 4, 15],
        [1800, 2, 30],
        [2200, 3, 25]
    ])
    
    y = np.array([460, 540, 330, 400])  # house prices in $1000s
    
    print("Dataset:")
    print("Feature matrix X (square footage, bedrooms, age):")
    for i, (sqft, beds, age) in enumerate(X):
        print(f"   House {i+1}: {sqft:4d} sqft, {beds} bedrooms, {age:2d} years old")
    
    print(f"\nTarget vector y (house prices in $1000s):")
    for i, price in enumerate(y):
        print(f"   House {i+1}: ${price}k")
    
    # 1. Set up the system as a least-squares problem Xβ = y
    print(f"\n1. Setting up least-squares problem Xβ = y:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   β will have shape: ({X.shape[1]},)")
    print(f"   β represents coefficients for [square_footage, bedrooms, age]")
    
    # 2. Solve for β using scipy.linalg.lstsq
    print(f"\n2. Solving for β using scipy.linalg.lstsq:")
    beta_lstsq, residuals, rank, s = lstsq(X, y)
    
    print(f"   Coefficients (β):")
    print(f"   - Square footage coefficient: {beta_lstsq[0]:10.6f} ($/sqft)")
    print(f"   - Bedrooms coefficient:       {beta_lstsq[1]:10.6f} ($/bedroom)")
    print(f"   - Age coefficient:            {beta_lstsq[2]:10.6f} ($/year)")
    
    # Handle residuals properly (can be scalar or array)
    if hasattr(residuals, '__len__') and len(residuals) > 0:
        print(f"   Residual sum of squares: {residuals[0]:.6f}")
    elif isinstance(residuals, (int, float)):
        print(f"   Residual sum of squares: {residuals:.6f}")
    
    print(f"   Matrix rank: {rank}")
    print(f"   Singular values: {s}")
    
    # 3. Use the resulting model to predict price of new house
    print(f"\n3. Using model to predict house price:")
    new_house = np.array([2400, 3, 20])  # 2400 sqft, 3 bedrooms, 20 years old
    predicted_price = new_house @ beta_lstsq
    
    print(f"   New house features: {new_house[0]} sqft, {new_house[1]} bedrooms, {new_house[2]} years old")
    print(f"   Predicted price: ${predicted_price:.2f}k")
    
    # Break down the prediction
    print(f"   Prediction breakdown:")
    print(f"   - Square footage contribution: {new_house[0]} × {beta_lstsq[0]:.6f} = ${new_house[0] * beta_lstsq[0]:.2f}k")
    print(f"   - Bedrooms contribution:       {new_house[1]} × {beta_lstsq[1]:.6f} = ${new_house[1] * beta_lstsq[1]:.2f}k")
    print(f"   - Age contribution:            {new_house[2]} × {beta_lstsq[2]:.6f} = ${new_house[2] * beta_lstsq[2]:.2f}k")
    print(f"   - Total prediction:                                    = ${predicted_price:.2f}k")
    
    # Evaluate model performance on training data
    y_pred_train = X @ beta_lstsq
    mse = np.mean((y - y_pred_train)**2)
    rmse = np.sqrt(mse)
    
    print(f"\n   Training set performance:")
    print(f"   Actual vs Predicted prices:")
    for i in range(len(y)):
        print(f"   House {i+1}: ${y[i]:3d}k (actual) vs ${y_pred_train[i]:6.1f}k (predicted)")
    print(f"   Mean Squared Error (MSE):  {mse:.6f}")
    print(f"   Root Mean Squared Error:   {rmse:.6f}")
    
    # 4. Compare with alternative method: direct solution via scipy.linalg.solve
    print(f"\n4. Comparing with alternative method (normal equations):")
    try:
        # For overdetermined system, use normal equations: X^T X β = X^T y
        XTX = X.T @ X
        XTy = X.T @ y
        
        print(f"   Using normal equations: (X^T X)β = X^T y")
        print(f"   X^T X shape: {XTX.shape}")
        print(f"   X^T y shape: {XTy.shape}")
        
        # Check condition number
        cond_num = np.linalg.cond(XTX)
        print(f"   Condition number of X^T X: {cond_num:.2e}")
        
        beta_solve = solve(XTX, XTy)
        
        print(f"   Coefficients from normal equations:")
        print(f"   - Square footage coefficient: {beta_solve[0]:10.6f}")
        print(f"   - Bedrooms coefficient:       {beta_solve[1]:10.6f}")
        print(f"   - Age coefficient:            {beta_solve[2]:10.6f}")
        
        predicted_price_solve = new_house @ beta_solve
        print(f"   Prediction using solve method: ${predicted_price_solve:.2f}k")
        
        print(f"\n   Comparison of methods:")
        diff = np.abs(beta_lstsq - beta_solve)
        print(f"   Absolute differences in coefficients:")
        print(f"   - Square footage: {diff[0]:.2e}")
        print(f"   - Bedrooms:       {diff[1]:.2e}")
        print(f"   - Age:            {diff[2]:.2e}")
        print(f"   Maximum difference: {np.max(diff):.2e}")
        
        if np.max(diff) < 1e-10:
            print(f"   ✓ Results are essentially identical (differences < 1e-10)")
        else:
            print(f"   ⚠ Results show some numerical differences")
        
    except np.linalg.LinAlgError as e:
        print(f"   ❌ Normal equations method failed: {e}")
        print(f"   This indicates the matrix X^T X is singular or ill-conditioned")
    
    
    return beta_lstsq, predicted_price

if __name__ == "__main__":
    coefficients, prediction = main()
    print(f"\nFinal Results:")
    print(f"Model coefficients: {coefficients}")
    print(f"Predicted price for new house: ${prediction:.2f}k")