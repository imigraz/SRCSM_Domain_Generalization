import os
from bin.source_matching.utils.dataset import Dataset


def compute_and_save_average_cdf(base_input_path):

    mmwhs_4labels_mr_dataset_train = Dataset(
        base_dataset_path=os.path.join(base_input_path, 'mmwhs17_4labels_mr'),
        modality='mr',
        image_suffix='_image',
        image_ext='.nii.gz',
        shortname='mmwhs17_4labels_mr',
        id_filename_subpath='cv/1/train.txt',
        image_folder='images',
        modality_file_suffix='',
    )
    mmwhs_4labels_mr_dataset_train.process_data_and_compute_average_cdf(save_average_cdf=True)

    return


def main():
    in_base_path = '/media0/franz/datasets/public_github_TEST/heart'

    compute_and_save_average_cdf(in_base_path)


if __name__ == '__main__':
    main()

