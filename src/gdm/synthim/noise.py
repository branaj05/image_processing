import numpy as np
def gaussian_noise(shape, mean=0, std=1):
    """
    Add Gaussian noise to an image.

    Parameters:
    shape (tuple): Shape of the image to be generated.
    mean (float): Mean of the Gaussian distribution.
    std (float): Standard deviation of the Gaussian distribution.

    Returns:
    np.ndarray: Noisy image with added Gaussian noise.
    """
    noise = np.random.normal(mean, std, shape)
    return noise