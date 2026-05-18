import os
import json


class AverageCDF:
    def __init__(self,
                 base_dataset_path,
                 shortname,
                 setup_folder='setup',
                 ):

        self.base_dataset_path = base_dataset_path
        self.shortname = shortname
        self.setup_path = os.path.join(self.base_dataset_path, setup_folder)
        self.average_cdf, self.overall_min_val, self.overall_max_val = self.load_average_cdf()

    def load_average_cdf(self):
        with open(os.path.join(self.setup_path, 'average_cdf.json'), 'r') as file:
            load_dict = json.load(file)
        average_cdf = load_dict['average_cdf']
        overall_min_val = load_dict['overall_min_val']
        overall_max_val = load_dict['overall_max_val']
        return average_cdf, overall_min_val, overall_max_val
