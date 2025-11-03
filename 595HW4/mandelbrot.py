import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_set(xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, width=800, height=600, max_iter=50, threshold=2):
    """
    Compute the Mandelbrot set over a specified range.
    
    Parameters:
    xmin, xmax: real range for the complex plane
    ymin, ymax: imaginary range for the complex plane
    width, height: resolution of the output image
    max_iter: maximum number of iterations
    threshold: threshold for determining if point escapes to infinity
    
    Returns:
    mask: boolean array indicating which points are in the Mandelbrot set
    """
    # Create coordinate arrays
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    
    # Create complex grid c = x + iy
    c = X + 1j * Y
    
    # Initialize z array (z0 = 0)
    z = np.zeros_like(c)
    
    # Initialize mask to track which points are in the set
    mask = np.zeros(c.shape, dtype=bool)
    
    # Iterate the Mandelbrot formula: z_{n+1} = z_n^2 + c
    for i in range(max_iter):
        # Only update points that haven't escaped yet
        not_escaped = np.abs(z) <= threshold
        
        # Update z for points that haven't escaped
        z[not_escaped] = z[not_escaped]**2 + c[not_escaped]
        
        # To avoid overflow, set z to large value for points that have escaped
        escaped = np.abs(z) > threshold
        z[escaped] = threshold + 1
    
    # Points are in the Mandelbrot set if they didn't escape
    mask = np.abs(z) <= threshold
    
    return mask

def plot_mandelbrot():
    """
    Generate and plot the Mandelbrot set fractal.
    """
    # Compute the Mandelbrot set
    print("Computing Mandelbrot set...")
    mask = mandelbrot_set()
    
    # Create the plot
    plt.figure(figsize=(12, 9))
    plt.imshow(mask, extent=[-2, 1, -1.5, 1.5], cmap='gray', origin='lower')
    plt.title('Mandelbrot Set')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    
    # Save the plot
    plt.savefig('mandelbrot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Mandelbrot set visualization saved as 'mandelbrot.png'")

if __name__ == "__main__":
    plot_mandelbrot()