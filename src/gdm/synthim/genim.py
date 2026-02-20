"""
Generate synthetic test imagery 
"""
#%% Import necessary libraries
from gdm.synthim.psf import gaussian_psf
from gdm.synthim.noise import gaussian_noise
from gdm.synthim.utils import timer, add_patch, rarg
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
# TODO: REMOVE AFTER DEV
from gdm.synthim.utils import *

#%% Functions and methods
@timer
def gen_psf(
        N,              
        size=(15, 15),  
        source_sigma=1.0,       
    ):
    """
    Parameters:
    N (int): Number of synthetic images to generate.
    size (int): The size of each synthetic image (size x size).
    source_sigma (float): The standard deviation of the Gaussian distribution for the PSF.
    snr (float): The signal-to-noise ratio (SNR) for noise level calculation.
    noise_mean (float): The mean of the Gaussian noise to be added to each image.
    noise_std (float): The standard deviation of the Gaussian noise to be added to each image.
    """
    h, w = size[0], size[1]
    # Random Pixel Locations
    dr, dc = np.random.uniform(-0.5, 0.5, N), np.random.uniform(-0.5, 0.5, N)  # Random sub-pixel shifts in [-0.5, 0.5]
    # Generate PSF Kernels
    psf = np.array([gaussian_psf(size, source_sigma, dr, dc, norm=np.max) for dr, dc in zip(dr, dc)])
    
    # rtest, ctest = 3.3, 3.0
    # noise_test = gaussian_noise(size, mean=0, std=1)
    # # noise_test = np.zeros(size)  # Disable noise for testing
    # psf_test = gaussian_psf(size, source_sigma, rtest-int(rtest), ctest-int(ctest), norm=np.max)
    # plt.figure(); imshow(add_patch(noise_test, psf_test*100, int(rtest), int(ctest))); plt.scatter(ctest, rtest)

    return psf, dr, dc

@timer
def gen_image_stack(
        num_images,                          # Number of synthetic images to generate
        size=(7, 7),                       # Size of each synthetic image (H, W)
        kernel=(17, 17),                     # PSF Kernel Size (h, w)
        source_sigma=1.0,                    # PSF Sigma
        A0=100,                              # Amplitude of the point sources - wtih noise_std = 1, A0 = SNR for single sources
        noise_mean=0,                        # Offset
        noise_std=1,                         # If None, calculated based on SNR
    ):
    """
    Generate a stack of synthetic images containing Gaussian point sources
    embedded in additive Gaussian noise.

    Each image contains either a single source or multiple sources,
    with subpixel positioning handled via fractional offsets in the
    generated PSF kernels.

    Parameters
    ----------
    num_images : int
        Number of synthetic images to generate.
    size : tuple of int, optional
        Shape of each generated image as (H, W). Default is (15, 15).
    kernel : tuple of int, optional
        Size of the PSF kernel as (h, w). Default is (15, 15).
    source_sigma : float, optional
        Standard deviation (in pixels) of the Gaussian PSF. Default is 1.0.
    A0 : float, optional
        Amplitude of each point source. If ``noise_std = 1``, this
        corresponds approximately to the signal-to-noise ratio (SNR)
        for single-source images.
    noise_mean : float, optional
        Mean of the additive Gaussian noise. Default is 0.
    noise_std : float, optional
        Standard deviation of the additive Gaussian noise.
        If None, the noise level may be computed from SNR assumptions.

    Returns
    -------
    images : np.ndarray
        Array of shape (num_images, H, W) containing generated images.
    r : np.ndarray
        Row coordinates (float, subpixel) of generated sources.
    c : np.ndarray
        Column coordinates (float, subpixel) of generated sources.
    A : np.ndarray
        Amplitudes of generated sources.

    Notes
    -----
    Source positions are generated uniformly within valid image bounds
    such that PSF kernels do not exceed the image edges. Subpixel shifts
    are handled by offsetting the Gaussian kernel prior to insertion.
        
    TODO: BIG - Refactor into a class complete with visualizations and testing
    """

    # Generate PSF Kernels and Source Parameters
    psf, dr, dc = gen_psf(num_images*2, kernel, source_sigma)
    psf*= A0  # Scale PSF by amplitude
    sep = source_sigma*5
    r, c = np.random.uniform(size[0]//2 - sep/2, size[0]//2 + sep/2, num_images*2), np.random.uniform(size[1]//2 - sep/2, size[1]//2 + sep/2, num_images*2)  # Ensure sources are well-separated and within bounds

    r0, c0 = np.round(r).astype(int), np.round(c).astype(int) # PSF center pixel coordinates
    # Generate Noise
    #    Noise Level Calculation based on SNR
    noise = np.array([gaussian_noise(size, mean=noise_mean, std=noise_std) for _ in range(num_images*2)])
    # Generate Images by adding PSF patches to noise background
    #   Randomly select num_images from the generated PSFs
    args = np.arange(num_images*2)
    sargs = rarg(args, size=num_images, replace=False)
    dargs = np.setdiff1d(args, sargs)  # Remaining indices for double sources
    #   Create single and double source images
    image = noise.copy()  # Start with noise background
    singles = np.array([add_patch(image[i, :, :], psf[i, :, :], r0[i], c0[i]) for i in sargs])
    doubles = np.array([add_patch(        single, psf[j, :, :], r0[j], c0[j]) for single, j in zip(singles.copy(), dargs)])

    # i = 5; plt.figure(); imshow(singles[i]); plt.scatter(c[sargs[i]], r[sargs[i]])
    # i = 5; plt.figure(); imshow(doubles[i]); plt.scatter(c[sargs[i]], r[sargs[i]]); plt.scatter(c[dargs[i]], r[dargs[i]])

    # TODO: Figure out return values and storage format
    return_vars ={'psf': psf, 'noise': noise, 'A': A0, 'r': r, 'c': c, 'singles': singles, 'doubles': doubles}
    return return_vars

if __name__ == "__main__":
    # psf = gen_psf(N=10, size=(15, 15), source_sigma=1.0, A0=1.0)
    img = gen_image_stack(num_images=10, size=(15, 15), 
                          source_sigma=1.0, 
                          A0=50, 
                          noise_mean=0, 
                          noise_std=1)

    a = 1

# %%
