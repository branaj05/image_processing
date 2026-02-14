"""
Generate synthetic test imagery 
"""
#%% Import necessary libraries
from gdm.synthim.psf import gaussian_psf
from gdm.synthim.noise import gaussian_noise
from matplotlib import image, pyplot as plt
import numpy as np
import h5py as h5

import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper
#%% Functions and methods
@timer
def gen(N, size, sigma):
    """
    Parameters:
    N (int): Number of synthetic images to generate.
    size (int): The size of each synthetic image (size x size).
    sigma (float): The standard deviation of the Gaussian distribution for the PSF.
    """
    images = []
    h, w = size[0], size[1]
    # Generate random pixel locations for N point sources, 
    # capturing at least 1 sigma of the energy
    # A = np.random.uniform(0.5, 1.5, N)  # Random amplitudes for the point sources
    A = 10*np.full(N)  # Uniform amplitudes for the point sources
    r, c = np.random.uniform(sigma, h-sigma, N), np.random.uniform(sigma, w-sigma, N)
    dr, dc = r-np.floor(r), c-np.floor(c)  # Sub-pixel shifts
    psf = np.array([Ai * gaussian_psf(size, sigma, dr, dc) for _, (dr, dc, Ai) in enumerate(zip(dr, dc, A))])
    noise = np.array([gaussian_noise(size, mean=0, std=1) for _ in range(N)])
    

    X, Y = np.meshgrid(np.arange(size[1]) - size[1]//2, np.arange(size[0]) - size[0]//2)
    r_idx = np.array([np.clip(Y + ri, 0, h-1) for ri in np.floor(r)], dtype=int)
    c_idx = np.array([np.clip(X + ci, 0, w-1) for ci in np.floor(c)], dtype=int)
    image = np.array([psf_i[r_idx_i, c_idx_i] + 0 
             for psf_i, noise_i, r_idx_i, c_idx_i in zip(psf, noise, r_idx, c_idx)])

    idx = np.array([2, 5, 6])
    idx = np.arange(N)
    idx = np.array([2, 6])
    plt.plot(r[idx], c[idx], 'ro', label='Point Sources')
    plt.imshow(np.sum([image[i, :, :] for i in idx], axis=0), cmap='viridis')
    plt.colorbar()  


    # 


if __name__ == "__main__":
    gen(N=10, size=(15, 15), sigma=1.0)
# %%
