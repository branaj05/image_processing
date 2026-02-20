import numpy as np
import time
#%% Utility functions
def rarg(arr, size, replace):
    rng = np.random.default_rng()
    return rng.choice(arr, size=size, replace=replace)

def add_patch(big: np.ndarray,
              small: np.ndarray,
              r: int,
              c: int) -> np.ndarray:
    """
    Insert a smaller 2D array into a larger 2D array centered at (r, c).

    Parameters
    ----------
    big : np.ndarray
        Destination image of shape (H, W).
    small : np.ndarray
        Patch image of shape (h, w).
    r : int
        Row index in `big` corresponding to the patch center.
    c : int
        Column index in `big` corresponding to the patch center.

    Returns
    -------
    np.ndarray
        Image with the patch inserted.
    """
    H, W = big.shape
    h, w = small.shape
    
    # Patch bounds in big
    r0 = r - h // 2
    c0 = c - w // 2
    
    r1 = r0 + h
    c1 = c0 + w
    
    # Clip to valid bounds
    br0 = max(r0, 0)
    bc0 = max(c0, 0)
    br1 = min(r1, H)
    bc1 = min(c1, W)
    
    # Corresponding slice in small image
    sr0 = br0 - r0
    sc0 = bc0 - c0
    sr1 = sr0 + (br1 - br0)
    sc1 = sc0 + (bc1 - bc0)
    
    big[br0:br1, bc0:bc1] += small[sr0:sr1, sc0:sc1]

    return big
#%% Mathematical functions
def center_of_mass(img: np.ndarray):
    Y, X = np.indices(img.shape)
    total = img.sum()
    r_com = (Y * img).sum() / total
    c_com = (X * img).sum() / total
    return r_com, c_com

#%% Visualization functions
def imshow(image, title=None):
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='viridis')
    if title:
        plt.title(title)
    plt.colorbar()
    plt.show()
#%% Housekeeping functions
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper