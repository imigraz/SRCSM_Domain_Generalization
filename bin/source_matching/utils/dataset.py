import os
import numpy as np
import json
from tqdm import tqdm
from bin.source_matching.utils.io_utils import get_file_ids
from bin.source_matching.utils.image import Image


class Dataset:
    def __init__(self,
                 base_dataset_path,
                 modality,
                 image_suffix,
                 image_ext,
                 shortname,
                 id_filename_subpath,
                 setup_folder='setup',
                 image_folder='images',
                 modality_file_suffix=None,
                 base_output_path='output/plots',
                 ):

        self.image_folder = image_folder
        self.modality = modality
        self.modality_file_suffix = modality_file_suffix

        self.base_dataset_path = base_dataset_path
        self.input_path = os.path.join(self.base_dataset_path, self.image_folder)
        self.setup_path = os.path.join(self.base_dataset_path, setup_folder)
        if self.modality_file_suffix is None:
            self.landmark_file_path = os.path.join(self.setup_path, 'landmark.csv')
        else:
            self.landmark_file_path = os.path.join(self.setup_path, f'{self.modality_file_suffix}landmark.csv')

        self.id_filename_path = os.path.join(self.setup_path, id_filename_subpath)

        self.file_ids = get_file_ids(self.id_filename_path)

        self.output_dir = base_output_path

        self.image_suffix = image_suffix
        self.image_ext = image_ext
        self.shortname = shortname

        self.modality = modality

        self.use_multiprocessing = False
        self.percentage_cpu_cores = 0.50  # might freeze if all cores are used

        self.image_dict = {cur_id: None for cur_id in self.file_ids}

    def process_data_and_compute_average_cdf(self, save_average_cdf=False):
        for cur_id in tqdm(self.file_ids, 'initialize image dict ...'):
            self.image_dict[cur_id] = self.load_image_with_id(cur_id)

        self.overall_min_val = int(np.min([x.min_val for x in self.image_dict.values()]))
        self.overall_max_val = int(np.max([x.max_val for x in self.image_dict.values()]))

        self.compute_sum_histogram()
        self.compute_average_cdf()

        if save_average_cdf:
            self.save_average_cdf()

    def compute_sum_histogram(self):
        values_hist_list = []

        for cur_id, cur_image in tqdm(self.image_dict.items(), 'compute average histogram ...'):
            values_hist_list.append(cur_image.histogram_in_mask)

        all_keys = sorted(np.unique(np.concatenate([list(cur_dict.keys()) for cur_dict in values_hist_list])))
        all_keys = [int(x) for x in all_keys]
        all_values_hist_dict = {k: 0 for k in all_keys}
        for cur_dict in values_hist_list:
            for k, v in cur_dict.items():
                all_values_hist_dict[k] += v

        self.average_histogram = all_values_hist_dict

    def compute_average_cdf(self):
        histogram = self.average_histogram

        value_sum = np.sum(list(histogram.values()))
        cumulative_sum = np.cumsum(list(histogram.values()))
        cumulative_sum_0_to_1 = [x / value_sum for x in cumulative_sum]

        x_values = list(histogram.keys())
        y_values = cumulative_sum_0_to_1

        values_hist = {x: y for x, y in zip(x_values, y_values)}
        self.average_cdf = values_hist

    def save_average_cdf(self):
        with open(os.path.join(self.setup_path, 'average_cdf.json'), 'w') as file:
            write_dict = {
                'average_cdf': self.average_cdf,
                'overall_min_val': self.overall_min_val,
                'overall_max_val': self.overall_max_val,
            }
            json.dump(write_dict, file)
        return

    def load_image_with_id(self, cur_id):
        image = Image(
            file_id=cur_id,
            modality=self.modality,
            image_path=self.input_path,
            file_name=f'{cur_id}{self.image_suffix}{self.image_ext}',
        )
        return image
