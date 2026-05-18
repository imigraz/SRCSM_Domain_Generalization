
import SimpleITK as sitk
from transformations.intensity.np.smooth import gaussian as gaussian_np
import numpy as np


def gaussian(image, sigma):
    """
    Multidimensional gaussian smoothing.
    Requires image size of at least 4 pixel in each dimension and sigma is always set to a value for all dimensions!
    :param image: sitk image
    :param sigma: list of sigmas per dimension, or scalar for equal sigma in each dimension
    :return: smoothed sitk image
    """
    return sitk.SmoothingRecursiveGaussian(image, sigma)


def gaussian_sitk_np_wrapper(image, sigma):
    """
    Multidimensional gaussian smoothing.
    Workaround solution to fix an issue with sitk.SmoothingRecursiveGaussian(...) which requires image size of at least
    4 pixel in each dimension. This function uses a different implementation of gaussian smoothing to circumvent this.
    :param image: sitk image
    :param sigma: list of sigmas per dimension, or scalar for equal sigma in each dimension
    :return: smoothed sitk image
    """
    image_np = sitk.GetArrayFromImage(image)
    out_image_np = gaussian_np(image_np, sigma)
    # print('min', np.min(image), 'max', np.max(image))
    out_image = sitk.GetImageFromArray(out_image_np)
    out_image.CopyInformation(image)
    return out_image
