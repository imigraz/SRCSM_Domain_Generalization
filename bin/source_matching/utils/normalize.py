import numpy as np


def robust_min_max(img, consideration_factors=(0.1, 0.1)):
    """
    Calculate a robust min and max by using quantiles.
    :param img: The np array.
    :param consideration_factors: Tuple of consideration factors.
                                  The minimum will be the quantile at consideration_factors[0] * 0.5.
                                  The maximum will be the quantile at 1 - consideration_factors[1] * 0.5
    :return: The value tuple.
    """
    return robust_quantile_min_max(img, (consideration_factors[0] * 0.5, 1 - consideration_factors[1] * 0.5))


def robust_quantile_min_max(img, min_max_quantiles=(0.1, 0.9)):
    """
    Calculate a robust min and max by using quantiles.
    :param img: The np array.
    :param min_max_quantiles: Tuple of minimum and maximum quantile.
    :return: The value tuple.
    """
    return np.quantile(img, min_max_quantiles)


def scale(img, old_range, new_range):
    """
    Scale an image with by mapping an old range to a new_simple range.
    :param img: The np array.
    :param old_range: The range of the input.
    :param new_range: The new_simple range of the output.
    :return: The scaled np array.
    """
    if old_range[0] == old_range[1] or new_range[0] == new_range[1]:
        return img
    shift = -old_range[0] + new_range[0] * (old_range[1] - old_range[0]) / (new_range[1] - new_range[0])
    scale = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    return (img + shift) * scale


def scale_min_max(img, new_range):
    """
    Scale an image with by mapping the minimum and maximum to a new_simple range.
    :param img: The np array.
    :param new_range: The range of the output.
    :return: The scaled np array.
    """
    old_range = np.amin(img), np.amax(img)
    return scale(img, old_range, new_range)


def normalize_mr_robust(img, out_range=(-1, 1), consideration_factor=0.1):
    """
    Scale an image with by mapping 0 and the value at the given quantile to a new_simple range.
    :param img: The np array.
    :param out_range: The range of the output.
    :param consideration_factor: The quantile at 1 - consideration_factor * 0.5 will be used as the maximum value.
    :return: The scaled np array.
    """
    _, max = robust_min_max(img, (0, consideration_factor))
    old_range = (0, max)
    return scale(img, old_range, out_range)


def normalize(img, out_range=(-1, 1)):
    """
    Scale an image with by mapping the minimum and maximum to a new_simple range.
    :param img: The np array.
    :param out_range: The range of the output.
    :return: The scaled np array.
    """
    min_value = np.min(img)
    max_value = np.max(img)
    old_range = (min_value, max_value)
    return scale(img, old_range, out_range)


def normalize_robust(img, out_range=(-1, 1), consideration_factors=(0.1, 0.1)):
    """
    Scale an image with by mapping a robust minimum and maximum to a new_simple range.
    The robust minimum and maximum will be calculated with robust_min_max()
    :param img: The np array.
    :param out_range: The range of the output.
    :param consideration_factors: Tuple of consideration factors.
    :return: The scaled np array.
    """
    min_value, max_value = robust_min_max(img, consideration_factors)
    if max_value == min_value:
        # fix to prevent div by zero
        max_value = min_value + 1
    old_range = (min_value, max_value)
    return scale(img, old_range, out_range)


def list_of_values_in_mask(image, mask):
    image_flat = np.ndarray.flatten(image)
    mask_flat = np.ndarray.flatten(mask)
    values_in_mask = image_flat.compress(mask_flat)
    values_in_mask = np.sort(values_in_mask)
    return values_in_mask


def normalize_robust_excluding_bg(img, out_range=(-1, 1), consideration_factors=(0.1, 0.1), bg_value=0.0):
    """
    Scale an image with by mapping a robust minimum and maximum to a new_simple range.
    The robust minimum and maximum will be calculated with robust_min_max()
    :param img: The np array.
    :param out_range: The range of the output.
    :param consideration_factors: Tuple of consideration factors.
    :return: The scaled np array.
    """
    fg_mask = np.where(img != bg_value, 1, 0).astype(img.dtype)
    values_in_mask = list_of_values_in_mask(img, fg_mask)
    min_value, max_value = robust_min_max(values_in_mask, consideration_factors)
    if np.min(img) == bg_value:
        min_value = bg_value

    # min_value, max_value = robust_min_max(img, consideration_factors)
    if max_value == min_value:
        # fix to prevent div by zero
        max_value = min_value + 1
    old_range = (min_value, max_value)
    return scale(img, old_range, out_range)


def normalize_quantile(img, out_range=(-1, 1), min_max_quantiles=(0.1, 0.9)):
    """
    Scale an image with by mapping a robust minimum and maximum based on quantiles to a new_simple range.
    The robust minimum and maximum will be calculated with robust_quantile_min_max()
    :param img: The np array.
    :param out_range: The range of the output.
    :param consideration_factors: Tuple of consideration factors.
    :return: The scaled np array.
    """
    min_value, max_value = robust_quantile_min_max(img, min_max_quantiles)
    if max_value == min_value:
        # fix to prevent div by zero
        max_value = min_value + 1
    old_range = (min_value, max_value)
    return scale(img, old_range, out_range)


def normalize_zero_mean_unit_variance(img):
    """
    Scale an image such that the output has a zero mean and a unit variance.
    :param img: The np array.
    :return: The scaled np array.
    """
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std if std > 0 else img
