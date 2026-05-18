import os
from bin.source_matching.utils.dataset import Dataset


def compute_and_save_average_cdf(base_input_path):

    mmwhs_ct_dataset = Dataset(
        base_dataset_path=os.path.join(base_input_path, 'mmwhs17_ct'),
        modality='ct',
        image_suffix='_image',
        image_ext='.nii.gz',
        id_filename_subpath='train_all.txt',
        shortname='mmwhs17ct',
        image_folder='images',
        modality_file_suffix='',
    )
    mmwhs_ct_dataset.process_data_and_compute_average_cdf(save_average_cdf=True)

    mmwhs_mr_dataset = Dataset(
        base_dataset_path=os.path.join(base_input_path, 'mmwhs17_mr'),
        modality='mr',
        image_suffix='_image',
        image_ext='.nii.gz',
        shortname='mmwhs17mr',
        id_filename_subpath='train_all.txt',
        image_folder='images',
        modality_file_suffix='',
    )
    mmwhs_mr_dataset.process_data_and_compute_average_cdf(save_average_cdf=True)

    return


def main():
    in_base_path = '/media0/franz/datasets/public_github_TEST/heart'

    compute_and_save_average_cdf(in_base_path)


if __name__ == '__main__':
    main()

