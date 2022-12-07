import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.morphology import label
import math


class CartoonEffect:

    def apply(img, sig_b, sig_x, p, N, T):

        if img.ndim != 2 and img.ndim != 3:
            raise ValueError('This image type is not supported.')

        # grey image
        if img.ndim == 2:
            image = gaussian(img, sig_b)
        # RGB image
        else:
            image = gaussian(rgb2gray(img), sig_b)

        # set Imin and Imax
        I_min = np.min(image)
        I_max = np.max(image)

        # allocate array bins
        bins = np.zeros(N + 1)

        for i in range(N):
            bins[i] = (((I_max - I_min) / N) * i) + I_min

        # allocate L image
        L_image = np.zeros(image.shape, dtype=int)

        for i in range(N - 1):
            if i == N - 1:
                mask = image >= bins[i]
            else:
                mask = np.array((image >= bins[i]) & (image < bins[i + 1]))
        L_image[mask] = i

        # allocate S image
        S_image = np.ones(img.shape, dtype=int)

        if img.ndim == 3:
            Labeled_image = label(L_image)
            for i in range(255):
                Ln = Labeled_image == i
                Ln_stack = np.stack((Ln, Ln, Ln), axis=2)
                S_image[Ln] = np.mean(img, where=Ln_stack, axis=(0, 1))
        else:
            S_image = (1 / N) * S_image

        XDoG, DoG = X_DoG(img, sig_x, p, T)
        output = S_image * XDoG

        return output, XDoG, DoG, S_image


def _convolve(img, kernel):
    '''Convenience method around ndimage.convolve.

    This calls ndimage.convolve with the boundary setting set to 'nearest'.  It
    also performs some checks on the input image.

    Parameters
    ----------
    img : numpy.ndarray
        input image
    kernel : numpy.ndarray
        filter kernel

    Returns
    -------
    numpy.ndarray
        filter image

    Raises
    ------
    ValueError
        if the image is not greyscale
    TypeError
        if the image or filter kernel are not a floating point type
    '''
    if img.ndim != 2:
        raise ValueError('Only greyscale images are supported.')

    if img.dtype != np.float32 and img.dtype != np.float64:
        raise TypeError('Image must be floating point.')

    if kernel.dtype != np.float32 and img.dtype != np.float64:
        raise TypeError('Filter kernel must be floating point.')

    return ndimage.convolve(img, kernel, mode='nearest')


def gaussian(img, sigma):
    '''Filter an image using a Gaussian kernel.

    The Gaussian is implemented internally as a two-pass, separable kernel.

    Note
    ----
    The kernel is scaled to ensure that its values all sum up to '1'.  The
    slight truncation means that the filter values may not actually sum up to
    one.  The normalization ensures that this is consistent with the other
    low-pass filters in this assignment.

    Parameters
    ----------
    img : numpy.ndarray
        a greyscale image
    sigma : float
        the width of the Gaussian kernel; must be a positive, non-zero value

    Returns
    -------
    numpy.ndarray
        the Gaussian blurred image; the output will have the same type as the
        input

    Raises
    ------
    ValueError
        if the value of sigma is negative
    '''

    # Determine filter width by doing 3*2*sigma
    N = max(math.ceil(6 * sigma), 3)

    # Add 1 for even N since N must be odd for a center pixel
    if N % 2 == 0:
        N = N + 1

    # Vertical kernel
    gauss_kernel_y = np.ones((N, 1), dtype=np.float32)
    for i in range(0, N):
        gauss_kernel_y[i, 0] = (1 / math.sqrt(2 * math.pi * sigma ** 2)) * \
                               math.exp(-(i - math.floor(0.5 * N)) ** 2 / (2 * sigma ** 2))

    # Horizontal kernel
    gauss_kernel_x = np.transpose(gauss_kernel_y)

    # Get sum to divide convolution result by
    sum = (gauss_kernel_y * gauss_kernel_x).sum()

    # Convolve vertical spatial mask with the image
    conv_y = _convolve(img, gauss_kernel_y)

    # Convolve horizontal spatial mask with previous output
    gauss_kernel_result = _convolve(conv_y, gauss_kernel_x)

    # Divide result by mask sum
    return (gauss_kernel_result / sum)

    # raise NotImplementedError('Implement this function/method.')


def X_DoG(img, sig_x, p, T):
    # xdog- eXtended difference of gaussian filters - gaussian
    # filter is basically a smoothing/blurring technique controlled by sigma, lower sigma

    if img.ndim != 2 and img.ndim != 3:
        raise ValueError('This image type is not supported.')

    # RGB image
    if img.ndim == 3:
        image = rgb2gray(img)
    # grey
    else:
        image = img

    # take Gaussian
    gauss_image = gaussian(image, sig_x)

    # find difference of gaussian
    DoG = gauss_image - gaussian(image, 1.6 * sig_x)

    U_image = gauss_image + (p * DoG)

    # Thresholding
    T = float(T)
    for j, k in np.ndindex(U_image.shape):
        if (U_image[j][k] > T):
            U_image[j][k] = 1
        else:
            U_image[j][k] = 0

    T_cleaned = LineCleanup(U_image)
    I_XDoG = np.stack((T_cleaned, T_cleaned, T_cleaned), axis=2)

    return I_XDoG, DoG


def LineCleanup(img):
    E_horz = _convolve(img, np.array([[1, 0, -1]]))
    E_vert = _convolve(img, np.array([[1, 0, -1]]).T)

    C_image = np.array((E_horz > 0) & (E_vert > 0))
    B_image = _convolve(img, (np.ones((3, 3))) / 9)

    I_filtered = np.copy(img)
    print(I_filtered)
    I_filtered[C_image] = B_image[C_image]
    print(I_filtered)
    return I_filtered
