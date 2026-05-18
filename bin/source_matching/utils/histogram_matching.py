import os
from multiprocessing import Pool
import bisect

import SimpleITK as sitk
import numpy as np
import pandas as pd
from bin.source_matching.utils.io_utils import create_dir


def list_of_values_in_mask(image, mask):
    image_flat = np.ndarray.flatten(image)
    mask_flat = np.ndarray.flatten(mask)
    values_in_mask = image_flat.compress(mask_flat)
    values_in_mask = np.sort(values_in_mask)
    return values_in_mask


def y_values_0_to_1(values_in_mask):
    return np.arange(0, len(values_in_mask)) / (len(values_in_mask) - 1)


def dict_accumulated_y_values_0_to_1(all_dest_values_hist_dict):
    value_sum = np.sum(list(all_dest_values_hist_dict.values()))
    cumulative_sum = np.cumsum(list(all_dest_values_hist_dict.values()))
    cumulative_sum_0_to_1 = [x / value_sum for x in cumulative_sum]
    return cumulative_sum_0_to_1


def unique_x_and_y(values_in_mask, y_values):
    x_values_unique, indices_unique = np.unique(values_in_mask, return_index=True)
    y_values_unique = y_values[indices_unique]
    return x_values_unique, y_values_unique


def get_physical_bbox(image, image_sitk, file_id, image_extent, keep_bbox_percentage=1, use_landmark_for_bbox=False, landmark_dict=None, landmark_file_path=None):
    image_shape = np.array(image.shape)
    if use_landmark_for_bbox and landmark_file_path is not None:
        physical_center_point = landmark_dict[file_id]
    else:
        physical_center_point = list(image_sitk.TransformContinuousIndexToPhysicalPoint([x / 2 for x in np.flip(image_shape)]))

    image_extent_to_use = np.array([x * keep_bbox_percentage for x in image_extent])
    physical_min = [center - (extent / 2) for center, extent in zip(physical_center_point, np.flip(image_extent_to_use))]
    physical_max = [center + (extent / 2) for center, extent in zip(physical_center_point, np.flip(image_extent_to_use))]
    bbox_min_orig = np.array((image_sitk.TransformPhysicalPointToIndex(physical_min)))
    bbox_max_orig = np.array((image_sitk.TransformPhysicalPointToIndex(physical_max)))

    # switch bbox min max indices in case of negative directions
    bbox_min = [min(x, y) for x, y in zip(bbox_min_orig, bbox_max_orig)]
    bbox_max = [max(x, y) for x, y in zip(bbox_min_orig, bbox_max_orig)]

    bbox_min = np.flip(bbox_min)
    bbox_max = np.flip(bbox_max)

    # if index is outside of the field of view, set index to the boundary
    bbox_min = np.where(bbox_min < 0, 0, bbox_min)
    bbox_max = np.where(bbox_max > image_shape, image_shape, bbox_max)

    # if index is on the boundary, move it x slices further into the bbox
    border_slice_offset = 2
    bbox_min = np.where(bbox_min < border_slice_offset, border_slice_offset, bbox_min)
    bbox_max = np.where(bbox_max > image_shape - border_slice_offset, image_shape - border_slice_offset, bbox_max)

    image_inner_bbox = image[
                            bbox_min[0]:bbox_max[0],
                            bbox_min[1]:bbox_max[1],
                            bbox_min[2]:bbox_max[2],
                            ]
    return image_inner_bbox


def get_center_bbox_using_slices(image, keep_bbox_percentage=1):
    keep_bbox_lower_upper = [(1 - keep_bbox_percentage) / 2, 1 - (1 - keep_bbox_percentage) / 2]
    bbox_indices = [[int(np.floor(x * keep_bbox_lower_upper[0])), int(np.ceil(x * keep_bbox_lower_upper[1]))] for x in
                    image.shape]
    image_inner_bbox = image[
                            bbox_indices[0][0]:bbox_indices[0][1],
                            bbox_indices[1][0]:bbox_indices[1][1],
                            bbox_indices[2][0]:bbox_indices[2][1],
                            ]
    return image_inner_bbox


def get_landmark_dict(landmark_file_path):
    landmark_data = pd.read_csv(landmark_file_path, header=None)
    landmark_dict = {x[0]: x[1:] for x in landmark_data.values}
    return landmark_dict


def compute_src_to_dest_dict(src_dict, dest_dict):
    src_to_dest_dict = {}

    dest_values = list(dest_dict.values())
    dest_keys = list(dest_dict.keys())
    for src_k, src_v in src_dict.items():
        idx = bisect.bisect_left(dest_values, src_v)
        if idx < len(dest_values) and dest_values[idx] == src_v:
            # Exact match
            cur_dest_k = dest_keys[idx]
        elif idx == 0:
            # src_v is smaller than any dest_v
            cur_dest_k = None  # or handle as needed
        else:
            # Use previous (lesser) value
            cur_dest_k = dest_keys[idx - 1]

        src_to_dest_dict[src_k] = cur_dest_k

    return src_to_dest_dict


def replace_values_with_closest_key(src_image_orig, src_to_dest_dict):
    image_vec = src_image_orig.flatten()
    src_to_dest_keys = np.array(list(src_to_dest_dict.keys()))  # src_image_inner_bbox
    idx = np.searchsorted(src_to_dest_keys, image_vec)
    msk = idx > len(src_to_dest_keys) - 1
    idx[msk] = len(src_to_dest_keys) - 1
    idx_new = np.array([idx[i] - 1 if abs(src_to_dest_keys[idx[i] - 1] - image_vec[i]) < abs(src_to_dest_keys[idx[i]] - image_vec[i]) else idx[i] for i in range(len(idx))])
    image_vec = np.array([src_to_dest_keys[_] for _ in idx_new])

    src_image = image_vec.reshape(src_image_orig.shape)
    return src_image


def perform_histogram_matching_single(src_file_id, cur_src_image, src_parameter_dataset, dst_parameter_dataset, output_path, output_dtype, output_ext, image_output_path=None):
    # load image if not yet loaded
    if cur_src_image == None:
        cur_src_image = src_parameter_dataset.load_image_with_id(src_file_id)

    src_shortname = src_parameter_dataset.shortname
    dst_shortname = dst_parameter_dataset.shortname

    if image_output_path == None:
        image_output_path = f'{output_path}/{src_shortname}/images_as_{dst_shortname}'

    src_dict = {cur_src_image.min_val: -1}
    src_dict.update(cur_src_image.cdf_in_mask)

    dst_dict = {dst_parameter_dataset.overall_min_val: -1}
    dst_dict.update(dst_parameter_dataset.average_cdf)

    src_to_dest_dict = compute_src_to_dest_dict(src_dict, dst_dict)

    # ------------------------------------------------------------
    # replace values with closest key to make sure that all values in image are defined in the translation dict
    print('replace_values_with_closest_key src_image_orig ...')

    image_sitk, image_np = cur_src_image.get_image_sitk_and_preprocessed_image_np()
    src_image_orig = image_np
    src_image_sitk = image_sitk

    src_image = replace_values_with_closest_key(src_image_orig, src_to_dest_dict)
    print('replace_values_with_closest_key src_image_orig finished')
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # translate image from src to dest
    print('applying src_to_dest_dict to src_image ...')
    src_image_as_dest = np.vectorize(src_to_dest_dict.__getitem__)(src_image)
    src_image_as_dest = src_image_as_dest.astype(output_dtype)
    print('src_to_dest_dict to src_image applied')
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # write image
    src_image_suffix = src_parameter_dataset.image_suffix
    src_as_dest_out_sitk = sitk.GetImageFromArray(src_image_as_dest)
    src_as_dest_out_sitk.SetOrigin(src_image_sitk.GetOrigin())
    src_as_dest_out_sitk.SetSpacing(src_image_sitk.GetSpacing())
    src_as_dest_out_sitk.SetDirection(src_image_sitk.GetDirection())
    converted_image_output_path = os.path.join(image_output_path)
    create_dir(converted_image_output_path)
    sitk.WriteImage(src_as_dest_out_sitk, os.path.join(converted_image_output_path, f'{src_file_id}{src_image_suffix}{output_ext}'), useCompression=True)
    # ------------------------------------------------------------


def perform_histogram_matching(src_parameter_dataset, dst_parameter_dataset, output_path, output_dtype, output_ext, use_multiprocessing, percentage_cpu_cores, image_output_path=None):

    # single processing
    if not use_multiprocessing:
        for src_file_id, cur_src_image in src_parameter_dataset.image_dict.items():
            perform_histogram_matching_single(src_file_id, cur_src_image, src_parameter_dataset, dst_parameter_dataset, output_path, output_dtype, output_ext, image_output_path)

    # multiprocessing
    else:
        with Pool(min(os.cpu_count(), int(os.cpu_count() * percentage_cpu_cores))) as pool:  # might freeze if all cores are used
            pool.starmap(perform_histogram_matching_single, [
                (src_file_id, cur_src_image, src_parameter_dataset, dst_parameter_dataset, output_path, output_dtype, output_ext, image_output_path)
                for src_file_id, cur_src_image in src_parameter_dataset.image_dict.items()])

    return

