import os
import numpy as np
from bin.source_matching.utils.histogram_matching import perform_histogram_matching
from bin.source_matching.utils.dataset import Dataset
from bin.source_matching.utils.average_cdf import AverageCDF


def match_bcv_to_rest(base_input_path, base_output_path, output_dtype, output_ext, use_multiprocessing, percentage_cpu_cores):
    bcv15_abdomen_4organs_test_dataset = Dataset(
        base_dataset_path=os.path.join(base_input_path, 'bcv15_abdomen_4organs_tr90'),
        modality='ct',
        image_suffix='_image',
        image_ext='.nii.gz',
        id_filename_subpath='test_all.txt',
        shortname='bcv15_abdomen_4organs_tr90',
    )

    chaos19_mr_t2_average_cdf = AverageCDF(
        base_dataset_path=os.path.join(base_input_path, 'chaos19_mr_t2_tr90'),
        shortname='chaos19_mr_t2_tr90',
    )

    average_cdf_list = [chaos19_mr_t2_average_cdf]
    for cur_average_cdf in average_cdf_list:
        perform_histogram_matching(
            src_parameter_dataset=bcv15_abdomen_4organs_test_dataset,
            dst_parameter_dataset=cur_average_cdf,
            output_path=base_output_path,
            output_dtype=output_dtype,
            output_ext=output_ext,
            use_multiprocessing=use_multiprocessing,
            percentage_cpu_cores=percentage_cpu_cores,
        )
    return


def match_chaos_to_rest(base_input_path, base_output_path, output_dtype, output_ext, use_multiprocessing, percentage_cpu_cores):
    chaos19_mr_t2_test_dataset = Dataset(
        base_dataset_path=os.path.join(base_input_path, 'chaos19_mr_t2_tr90'),
        modality='mr',
        image_suffix='_image',
        image_ext='.nii.gz',
        id_filename_subpath='test_all.txt',
        shortname='chaos19_mr_t2_tr90',
    )

    bcv15_abdomen_4organs_average_cdf = AverageCDF(
        base_dataset_path=os.path.join(base_input_path, 'bcv15_abdomen_4organs_tr90'),
        shortname='bcv15_abdomen_4organs_tr90',
    )

    average_cdf_list = [bcv15_abdomen_4organs_average_cdf]
    for cur_average_cdf in average_cdf_list:
        perform_histogram_matching(
            src_parameter_dataset=chaos19_mr_t2_test_dataset,
            dst_parameter_dataset=cur_average_cdf,
            output_path=base_output_path,
            output_dtype=output_dtype,
            output_ext=output_ext,
            use_multiprocessing=use_multiprocessing,
            percentage_cpu_cores=percentage_cpu_cores,
        )
    return


def main():
    in_base_path = '/media0/franz/datasets/public_github_TEST/multi_organ'
    out_base_path = in_base_path

    output_dtype = np.int16
    output_ext = '.nii.gz'
    use_multiprocessing = True
    percentage_cpu_cores = 0.5

    match_bcv_to_rest(in_base_path, out_base_path, output_dtype, output_ext, use_multiprocessing, percentage_cpu_cores)
    match_chaos_to_rest(in_base_path, out_base_path, output_dtype, output_ext, use_multiprocessing, percentage_cpu_cores)


if __name__ == '__main__':
    main()



