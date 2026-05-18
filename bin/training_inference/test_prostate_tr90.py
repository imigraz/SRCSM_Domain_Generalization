import os

from main import run_test


if __name__ == '__main__':

    # set according to dataset
    cur_cv = 0  # 0: uses all test domain data; 1: uses held-out test domain data used for in-domain comparisons

    # set setup folder to the respective training domain - inference will run on all other domains
    # Note: in this setup, a model trained on each domain is required for evaluation
    train_setup = 'setup_A' # 'setup_A', 'setup_B', 'setup_C', 'setup_D', 'setup_E', 'setup_F'
    setup_list = ['setup_A', 'setup_B', 'setup_C', 'setup_D', 'setup_E', 'setup_F']

    base_dataset_folder_inference = '/PATH/TO/datasets'
    base_output_folder = '/PATH/TO/experiments/prostate/prostate_tr90'

    dataset_name_train = 'prostate_tr90'
    train_modality = 'mr'
    num_labels = 2  # including background

    # image resampling parameters: lead to a spacing of 0.75, 0.75, 0.75
    image_size = [128, 128, 128]  # used in main paper
    # image_size = [96, 96, 96]  # can be used for faster templating with reduced memory footprint and computational cost
    image_extent = [96, 96, 96]

    is_inference = True

    # ------------------------------------------------------------
    # inference
    load_model_and_modality_list = [
        dict(
            load_model_base='/PATH/TO/TRAINED/MODEL/DATE/FOLDER',
            load_model_iter=50000,
            final_output_folder_in_path='modelA',
        ),
    ]

    wrapper_dict_list = []
    for cur_params_dict in load_model_and_modality_list:
        for test_setup in setup_list:
            if train_setup == test_setup:
                continue

            cur_wrapper_dict_list = [

                # --------------------------------------------------------
                # use original test domain data
                # dict(
                #     additional_dataset_parameters=dict(
                #         intensity_postprocessing='mr',
                #         image_base_folder='images',
                #         setup_folder_to_use=test_setup,
                #     ),
                #     dataset_folder_subpath='prostate/prostate_tr90',
                #     has_validation_groundtruth=True,
                #     dataset_name_train=f'{dataset_name_train}_{train_setup}',
                #     dataset_name_test=f'{dataset_name_train}_{test_setup}',
                #     modality='mr',
                # ),
                # --------------------------------------------------------

                # --------------------------------------------------------
                # use source matched test domain data
                dict(
                    additional_dataset_parameters=dict(
                        intensity_postprocessing=train_modality,
                        image_base_folder=f'images_as_{train_setup}',
                        setup_folder_to_use=test_setup,
                    ),
                    dataset_folder_subpath='prostate/prostate_tr90',
                    has_validation_groundtruth=True,
                    dataset_name_train=f'{dataset_name_train}_{train_setup}',
                    dataset_name_test=f'{dataset_name_train}_{test_setup}',
                    modality='mr',
                ),
                # --------------------------------------------------------

            ]
            wrapper_dict_list.extend(cur_wrapper_dict_list)

    # ------------------------------------------------------------

        for wrapper_dict in wrapper_dict_list:
            base_dataset_folder = os.path.join(base_dataset_folder_inference, wrapper_dict['dataset_folder_subpath'])

            run_test(
                cur_cv=cur_cv,
                setup=train_setup,
                base_dataset_folder=base_dataset_folder,
                base_output_folder=base_output_folder,
                num_labels=num_labels,
                image_size=image_size,
                image_extent=image_extent,
                is_inference=is_inference,
                params_dict=cur_params_dict,
                wrapper_dict=wrapper_dict,
            )

