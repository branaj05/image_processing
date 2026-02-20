#%%
#%%
import numpy as np
import matplotlib.pyplot as plt
def gaussian_psf(size, sigma, dr, dc, 
                 norm = np.max,           # [np.max | np.sum] Normalization method - configure for easy control of peak signal (max) or total energy (sum)
    ):
    """
    Generate an isotropic 2D Gaussian Point Spread Function (PSF).
    NOTE: This is not the most physically accurate PSF, 
          but it serves as a simple model for demonstration purposes.

    Parameters
    ----------
    size : tuple
        The size of the PSF Kernel (int, int).
    sigma : float 
        The standard deviation of the Gaussian distribution.

    Returns
    -------
    psf : np.ndarray: 
        A 2D array representing the Gaussian-distributed PSF.
    """
    r,c = size[0] // 2, size[1] // 2
    x = np.arange(size[0]) - r
    y = np.arange(size[1]) - c
    X, Y = np.meshgrid(x, y)
    psf = np.exp(-((X - dc) ** 2 + (Y - dr) ** 2) / (2 * sigma ** 2))
    psf = psf / norm(psf)  # Normalize the PSF to represent a fraction of total energy 
    
    return psf

if __name__ == "__main__":
    # define the amplitude (A) and the size of the PSF kernel (N)
    A = 1
    K = 8
    # define the sub-pixel shift (dx, dy) and the standard deviation (sig) for the Gaussian PSF
    dx, dy = 0.33, 0.33
    # approximate a point source as a Gaussian distribution with a standard deviation of 1 pixel
    sig = 1.0
    N = np.ceil(K * sig).astype(int)  # Ensure N is an integer and large enough to capture the PSF
    if N % 2 == 0: N+= 1  # Ensure N is odd to have a central pixel
    # Compute the PSF
    psf = A * gaussian_psf((N, N), sig, dy, dx)

    # Visualize the PSF
    plt.imshow(psf, cmap='viridis')
    plt.colorbar()
    plt.show()
