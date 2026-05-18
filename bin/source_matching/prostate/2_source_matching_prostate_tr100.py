import os
import numpy as np
from bin.source_matching.utils.histogram_matching import perform_histogram_matching
from bin.source_matching.utils.dataset import Dataset
from bin.source_matching.utils.average_cdf import AverageCDF


def match_prostate_tr100_to_rest(base_input_path, base_output_path, output_dtype, output_ext, use_multiprocessing, percentage_cpu_cores):

    all_setups = ['setup_A', 'setup_B', 'setup_C', 'setup_D', 'setup_E', 'setup_F']
    for test_setup in all_setups:
        for train_setup in all_setups:
            if train_setup == test_setup:
                continue

            prostate_dataset_test = Dataset(
                base_dataset_path=os.path.join(base_input_path, 'prostate_tr100'),
                modality='mr',
                image_suffix='_image',
                image_ext='.nii.gz',
                id_filename_subpath='test_all.txt',
                shortname=f'prostate_{test_setup}',
                setup_folder=test_setup,
                image_folder='images',
                modality_file_suffix='',
                base_output_path=os.path.join(base_output_path, 'plots'),
            )

            prostate_dataset_average_cdf = AverageCDF(
                base_dataset_path=os.path.join(base_input_path, 'prostate_tr100'),
                shortname=f'prostate_{train_setup}',
                setup_folder=train_setup,
            )

            average_cdf_list = [prostate_dataset_average_cdf]
            for cur_average_cdf in average_cdf_list:
                perform_histogram_matching(
                    src_parameter_dataset=prostate_dataset_test,
                    dst_parameter_dataset=cur_average_cdf,
                    output_path=base_output_path,
                    output_dtype=output_dtype,
                    output_ext=output_ext,
                    use_multiprocessing=use_multiprocessing,
                    percentage_cpu_cores=percentage_cpu_cores,
                    image_output_path=os.path.join(base_output_path, 'prostate_tr100', f'images_as_{train_setup}'),
                )
    return


def main():
    in_base_path = '/media0/franz/datasets/public_github_TEST/prostate'
    out_base_path = in_base_path

    output_dtype = np.int16
    output_ext = '.nii.gz'
    use_multiprocessing = False
    percentage_cpu_cores = 0.5

    match_prostate_tr100_to_rest(in_base_path, out_base_path, output_dtype, output_ext, use_multiprocessing, percentage_cpu_cores)


if __name__ == '__main__':
    main()



