import h5py as h5
from gdm.synthim.psf import gaussian_psf
import numpy as np

def gen_data(filename, num_samples=1000, sig=1.0):
    # Allocate image array
    
    # Define Random Pixel Locations

    # Define Random Amplitudes

    # Generate PSF Kernels



    return




if __name__ == "__main__":
    # define the amplitude (A) and the size of the PSF kernel (N)
    A = 1
    K = 8
    # define the sub-pixel shift (dx, dy) and the standard deviation (sig) for the Gaussian PSF
    dx, dy = 0.0, 0.0
    # approximate a point source as a Gaussian distribution with a standard deviation of 1 pixel
    sig = 1.0
    N = np.ceil(K * sig).astype(int)  # Ensure N is an integer and large enough to capture the PSF
    if N % 2 == 0: N+= 1  # Ensure N is odd to have a central pixel
    # Compute the PSF
    psf = A * gaussian_psf(N, sig, dx, dy)