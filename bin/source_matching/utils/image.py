import os
import SimpleITK as sitk
import numpy as np


class Image:
    def __init__(self,
                 file_id,
                 modality,
                 image_path,
                 file_name,
                 ):
        self.file_id = file_id
        self.modality = modality
        self.image_path = image_path
        self.file_name = file_name

        if self.modality == 'ct':
            self.lower_thres = -1023
            self.upper_thres = 1024
            self.upper_percentile_thres = None
            self.upper_normalize_to_value = None
        elif self.modality == 'mr':
            self.lower_thres = None
            self.upper_thres = None
            self.upper_percentile_thres = 100
            self.upper_normalize_to_value = 2048
        else:
            print(f'modality {modality} unknown. either ct or mr')
            exit(-1)

        self.use_mask_min_val = True
        self.use_mask_max_val = True

        # initialize some variables but do not save the images as members to save memory
        image_sitk, local_image_np = self.get_image_sitk_and_preprocessed_image_np()

    def get_image_sitk_and_preprocessed_image_np(self):
        image_sitk = self.get_image_sitk()
        local_image_np = sitk.GetArrayFromImage(image_sitk)

        tmp_image_np = np.copy(local_image_np)

        # ct norm
        if self.lower_thres:
            tmp_image_np = np.where(tmp_image_np < self.lower_thres, self.lower_thres, tmp_image_np)
        if self.upper_thres:
            tmp_image_np = np.where(tmp_image_np > self.upper_thres, self.upper_thres, tmp_image_np)
        # mr norm
        if self.upper_percentile_thres:
            upper_percentile_val = np.percentile(tmp_image_np, self.upper_percentile_thres)
            tmp_image_np = np.where(tmp_image_np > upper_percentile_val, upper_percentile_val, tmp_image_np)
        local_image_np = tmp_image_np

        if self.upper_normalize_to_value:
            cur_min_val = 0
            cur_max_val = np.max(local_image_np)
            new_min_val = 0
            new_max_val = self.upper_normalize_to_value
            local_image_np = (local_image_np - cur_min_val) * (new_max_val - new_min_val) / (
                        cur_max_val - cur_min_val) + new_min_val

        # round and convert to int
        local_image_np = np.round(local_image_np).astype(np.int32)

        self.mask = np.ones_like(local_image_np)
        if self.use_mask_min_val:
            min_val = np.min(local_image_np)
            self.mask = np.where(local_image_np == min_val, False, self.mask)
        if self.use_mask_max_val:
            max_val = np.max(local_image_np)
            self.mask = np.where(local_image_np == max_val, False, self.mask)

        self.src_values_in_mask = self.list_of_values_in_mask(local_image_np, self.mask)

        self.histogram_in_mask = self.compute_histogram(self.src_values_in_mask)
        self.cdf_in_mask = self.compute_cdf(self.histogram_in_mask)

        self.min_val = np.min(local_image_np)
        self.max_val = np.max(local_image_np)
        return image_sitk, local_image_np

    def get_image_sitk(self):
        image_sitk = sitk.ReadImage(os.path.join(self.image_path, self.file_name))
        return image_sitk

    def get_image_sitk_and_image_np(self):
        image_sitk = sitk.ReadImage(os.path.join(self.image_path, self.file_name))
        image_np = sitk.GetArrayFromImage(image_sitk)
        return image_sitk, image_np

    def compute_histogram(self, values):
        hist, bin_edges = np.histogram(values, sorted(np.unique(values)))
        values_hist = {y: x for x, y in zip(hist, bin_edges[:-1])}
        return values_hist

    def compute_cdf(self, histogram):
        value_sum = np.sum(list(histogram.values()))
        cumulative_sum = np.cumsum(list(histogram.values()))
        cumulative_sum_0_to_1 = [x / value_sum for x in cumulative_sum]

        x_values = list(histogram.keys())
        y_values = cumulative_sum_0_to_1

        values_hist = {x: y for x, y in zip(x_values, y_values)}
        return values_hist

    def list_of_values_in_mask(self, image, mask):
        image_flat = np.ndarray.flatten(image)
        mask_flat = np.ndarray.flatten(mask)
        values_in_mask = image_flat.compress(mask_flat)
        values_in_mask = np.sort(values_in_mask)
        return values_in_mask
