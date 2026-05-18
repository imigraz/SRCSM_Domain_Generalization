import os
import numpy as np
import csv


class TestTimeAdaptationLogger(object):
    def __init__(self, metric_names):
        self.metric_names = metric_names
        self.metrics_dict = {}

    def append_metric_values(self, metric_values, current_id, iter_str):
        iter_int = int(iter_str)
        if current_id not in self.metrics_dict:
            self.metrics_dict[current_id] = {}
        self.metrics_dict[current_id][iter_int] = metric_values
        return

    def compute_average_dict(self, metric_names_key):
        iter_keys = list(list(self.metrics_dict.values())[0].keys())
        average_dict = {}
        for cur_key in iter_keys:
            cur_metric_list = []
            for cur_id, sub_dict in self.metrics_dict.items():
                cur_metric_dict = sub_dict[cur_key]
                metric_values = cur_metric_dict[metric_names_key]
                cur_metric_list.append(metric_values)
            average = list(np.average(cur_metric_list, axis=0))
            average_dict[cur_key] = average
        return average_dict

    def compute_mean_matrix_dict(self, metric_names_key):
        iter_keys = list(list(self.metrics_dict.values())[0].keys())
        mean_matrix_dict = {}
        for cur_key in iter_keys:
            cur_mean_list = []
            for cur_id, sub_dict in self.metrics_dict.items():
                cur_metric_dict = sub_dict[cur_key]
                metric_values = cur_metric_dict[metric_names_key]
                cur_mean = np.mean(metric_values)
                cur_mean_list.append(cur_mean)
            mean_matrix_dict[cur_key] = cur_mean_list
        return mean_matrix_dict

    def write_output(self, output_path):
        for metric_names_key, metric_names_value in self.metric_names.items():
            average_dict = self.compute_average_dict(metric_names_key)
            with open(os.path.join(output_path, f'{metric_names_key}_mean_for_tta_iter.csv'), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(['tta_iter'] + metric_names_value)
                for k, v in average_dict.items():
                    writer.writerow([k] + [np.average(v)] + v)

            mean_matrix_dict = self.compute_mean_matrix_dict(metric_names_key)
            id_keys = list(self.metrics_dict.keys())
            with open(os.path.join(output_path, f'{metric_names_key}_mean_tta_matrix.csv'), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(['tta_iter'] + ['mean'] + id_keys)
                for k, v in mean_matrix_dict.items():
                    writer.writerow([k] + [np.average(v)] + v)
