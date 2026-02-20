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
    h, w = size[0], size[1]
    # Generate random pixel locations for N point sources, 
    # capturing at least 1 sigma of the energy
    # A = np.random.uniform(0.5, 1.5, N)  # Random amplitudes for the point sources
    A = 10*np.ones(N)  # Uniform amplitudes for the point sources
    r, c = np.random.uniform(sigma, h-sigma, N), np.random.uniform(sigma, w-sigma, N)
    dr, dc = r-np.round(r), c-np.round(c)  # Sub-pixel shifts
    psf = np.array([Ai * gaussian_psf(size, sigma, dc, dr) for _, (dr, dc, Ai) in enumerate(zip(dr, dc, A))])
    noise = np.array([gaussian_noise(size, mean=0, std=1) for _ in range(N)])
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    y = np.arange(size[0]) - size[0] // 2
    x = np.arange(size[1]) - size[1] // 2
    Y, X = np.meshgrid(y, x, indexing='ij')
    image = np.zeros((N, size[0], size[1]))
    for i in range(N):
        image[
            i, 
            np.clip(Y + r0[i], 0, h-1), 
            np.clip(X + c0[i], 0, w-1), 
        ] += psf[i]

    idx = np.array([2, 5, 6])
    # idx = np.arange(N)
    # idx = np.array([2, 6])
    plt.plot(c[idx],r[idx], 'ro', label='Point Sources')
    plt.imshow(np.sum([image[i, :, :] for i in idx], axis=0), cmap='viridis')
    # plt.imshow(np.sqrt(np.sum([noise[i, :, :]**2 for i in idx], axis=0)), cmap='viridis')
    plt.colorbar()  
    # 
    y = np.arange(size[0]) - size[0] // 2
    x = np.arange(size[1]) - size[1] // 2
    Y, X = np.meshgrid(y, x, indexing='ij')
    r, c = 14, 14
    dr = r - np.round(r)
    dc = c - np.round(c)
    psf = gaussian_psf(size, sigma, dc, dr)
    i0 = np.zeros(size)
    i0[np.clip(Y + int(r), 0, h-1), np.clip(X + int(c), 0, w-1)] = psf
    plt.figure()
    plt.imshow(i0, cmap='viridis')
    plt.scatter(c, r, color='red', label='Point Source')
    plt.colorbar()

if __name__ == "__main__":
    gen(N=10, size=(15, 15), sigma=1.0)
# %%
