#%%
import numpy as np
import matplotlib.pyplot as plt
def gaussian_psf(size, sigma, dx, dy):
    """
    Generate an isotropic 2D Gaussian Point Spread Function (PSF).
    NOTE: This is not the most physically accurate PSF, 
          but it serves as a simple model for demonstration purposes.

    Parameters:
    size (int): The size of the PSF Kernel (size x size). 
    sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
    np.ndarray: A 2D array representing the Gaussian-distributed PSF.
    """
    r = size // 2
    x = np.arange(size) - r
    y = np.arange(size) - r
    X, Y = np.meshgrid(x, y)
    psf = np.exp(-((X - dx) ** 2 + (Y - dy) ** 2) / (2 * sigma ** 2))
    psf = psf / np.sum(psf)  # Normalize the PSF to represent a fraction of total energy 
    return psf

if __name__ == "__main__":
    # define the amplitude (A) and the size of the PSF kernel (N)
    A = 1
    K = 8
    # define the sub-pixel shift (dx, dy) and the standard deviation (sig) for the Gaussian PSF
    dx, dy = 0.0, 0.0
    sig = 1.0
    N = np.ceil(K * sig).astype(int)  # Ensure N is an integer and large enough to capture the PSF
    if N % 2 == 0: N+= 1  # Ensure N is odd to have a central pixel
    # Compute the PSF
    psf = A * gaussian_psf(N, sig, dx, dy)

    # Visualize the PSF
    plt.imshow(psf, cmap='viridis')
    plt.colorbar()
