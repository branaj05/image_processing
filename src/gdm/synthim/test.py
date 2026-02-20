import numpy as np
from gdm.synthim.genim import gen_psf
from gdm.synthim.utils import add_patch, center_of_mass

def test_alignment():
    # TODO: Fold into genim.py after development, currently serves as a testbed for PSF generation and alignment
    size = (15, 15)
    sigma = 1.0
    A0 = 1000

    psf, r, c, r0, c0, A = gen_psf(1, size, sigma, A0)

    img = np.zeros(size)
    img = add_patch(img, psf[0], r0[0], c0[0])

    r_com, c_com = center_of_mass(img)

    print("Target:", r[0], c[0])
    print("COM   :", r_com, c_com)

    assert np.allclose([r_com, c_com], [r[0], c[0]], atol=1e-3)