import os
import numpy as np
from bin.source_matching.utils.histogram_matching import perform_histogram_matching
from bin.source_matching.utils.dataset import Dataset
from bin.source_matching.utils.average_cdf import AverageCDF


def match_mnms21_to_rest(base_input_path, base_output_path, output_dtype, output_ext, use_multiprocessing, percentage_cpu_cores):
    mnms21_sa_labeled_dataset_test = Dataset(
        base_dataset_path=os.path.join(base_input_path, 'mnms21_sa_labeled'),
        modality='mr',
        image_suffix='_image',
        image_ext='.nii.gz',
        shortname='mnms21_sa_labeled',
        id_filename_subpath='test_all.txt',
        image_folder='images',
        modality_file_suffix='',
    )

    mmwhs_mr_average_cdf = AverageCDF(
        base_dataset_path=os.path.join(base_input_path, 'mmwhs17_mr'),
        shortname='mmwhs17_mr',
    )

    mmwhs_ct_average_cdf = AverageCDF(
        base_dataset_path=os.path.join(base_input_path, 'mmwhs17_ct'),
        shortname='mmwhs17_ct',
    )

    average_cdf_list = [mmwhs_mr_average_cdf, mmwhs_ct_average_cdf]
    for cur_average_cdf in average_cdf_list:
        perform_histogram_matching(
            src_parameter_dataset=mnms21_sa_labeled_dataset_test,
            dst_parameter_dataset=cur_average_cdf,
            output_path=base_output_path,
            output_dtype=output_dtype,
            output_ext=output_ext,
            use_multiprocessing=use_multiprocessing,
            percentage_cpu_cores=percentage_cpu_cores,
        )
    return


def main():
    in_base_path = '/media0/franz/datasets/public_github_TEST/heart'
    out_base_path = in_base_path

    output_dtype = np.int16
    output_ext = '.nii.gz'
    use_multiprocessing = True
    percentage_cpu_cores = 0.5

    match_mnms21_to_rest(in_base_path, out_base_path, output_dtype, output_ext, use_multiprocessing, percentage_cpu_cores)


if __name__ == '__main__':
    main()



