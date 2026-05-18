import os
from bin.source_matching.utils.dataset import Dataset


def compute_and_save_average_cdf(base_input_path):

    bcv15_abdomen_4organs_train_dataset = Dataset(
        base_dataset_path=os.path.join(base_input_path, 'bcv15_abdomen_4organs_tr90'),
        modality='ct',
        image_suffix='_image',
        image_ext='.nii.gz',
        id_filename_subpath='cv/1/train.txt',
        shortname='bcv15_abdomen_4organs_tr90',
    )
    bcv15_abdomen_4organs_train_dataset.process_data_and_compute_average_cdf(save_average_cdf=True)

    chaos19_mr_t2_train_dataset = Dataset(
        base_dataset_path=os.path.join(base_input_path, 'chaos19_mr_t2_tr90'),
        modality='mr',
        image_suffix='_image',
        image_ext='.nii.gz',
        id_filename_subpath='cv/1/train.txt',
        shortname='chaos19_mr_t2_tr90',
    )
    chaos19_mr_t2_train_dataset.process_data_and_compute_average_cdf(save_average_cdf=True)

    return


def main():
    in_base_path = '/media0/franz/datasets/public_github_TEST/multi_organ'

    compute_and_save_average_cdf(in_base_path)


if __name__ == '__main__':
    main()

