import os
from bin.source_matching.utils.dataset import Dataset


def compute_and_save_average_cdf(base_input_path):
    all_setups = ['setup_A', 'setup_B', 'setup_C', 'setup_D', 'setup_E', 'setup_F']
    for train_setup in all_setups:

        prostate_dataset_train = Dataset(
            base_dataset_path=os.path.join(base_input_path, 'prostate_tr100'),
            modality='mr',
            image_suffix='_image',
            image_ext='.nii.gz',
            id_filename_subpath='train_all.txt',
            shortname=f'prostate_{train_setup}',
            setup_folder=train_setup,
            image_folder='images',
            modality_file_suffix='',
        )
        prostate_dataset_train.process_data_and_compute_average_cdf(save_average_cdf=True)

    return


def main():
    in_base_path = '/media0/franz/datasets/public_github_TEST/prostate'

    compute_and_save_average_cdf(in_base_path)


if __name__ == '__main__':
    main()

