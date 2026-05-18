import os

from main import run_test


if __name__ == '__main__':

    # set according to dataset
    cur_cv = 0  # 0: uses all test domain data; 1: uses only held-out test domain data used for in-domain comparisons
    setup = 'setup'

    base_dataset_folder_inference = '/PATH/TO/datasets'
    base_output_folder = '/PATH/TO/experiments/multi_organ/amos22_4organs_mr'

    dataset_name_train = 'amos22_4organs_mr'
    train_modality = 'mr'
    num_labels = 5  # including background

    # image resampling parameters: lead to a spacing of 2.4, 2.4, 2.4
    image_size = [160, 128, 160]  # used in main paper
    # image_size = [80, 64, 80]  # can be used for faster templating with reduced memory footprint and computational cost
    image_extent = [384, 307.2, 384]

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

    for cur_params_dict in load_model_and_modality_list:

        wrapper_dict_list = [

            # --------------------------------------------------------
            # use original test domain data
            # dict(
            #     additional_dataset_parameters=dict(
            #         intensity_postprocessing='ct',
            #         image_base_folder='images',
            #     ),
            #     dataset_folder_subpath='multi_organ/bcv15_abdomen_4organs',
            #     has_validation_groundtruth=True,
            #     dataset_name_train=dataset_name_train,
            #     dataset_name_test='bcv15_abdomen_4organs',
            #     modality='ct',
            # ),
            # # --------------------------------------------------------
            # # use original test domain data
            # dict(
            #     additional_dataset_parameters=dict(
            #         intensity_postprocessing='mr',
            #         image_base_folder='images',
            #     ),
            #     dataset_folder_subpath='multi_organ/chaos19_mr_t2',
            #     has_validation_groundtruth=True,
            #     dataset_name_train=dataset_name_train,
            #     dataset_name_test='chaos19_mr_t2',
            #     modality='mr',
            # ),
            # # --------------------------------------------------------
            # # use original test domain data
            # dict(
            #     additional_dataset_parameters=dict(
            #         intensity_postprocessing='ct',
            #         image_base_folder='images',
            #     ),
            #     dataset_folder_subpath='multi_organ/amos22_4organs_ct',
            #     has_validation_groundtruth=True,
            #     dataset_name_train=dataset_name_train,
            #     dataset_name_test='amos22_4organs_ct',
            #     modality='ct',
            # ),
            # --------------------------------------------------------


            # --------------------------------------------------------
            # use source matched test domain data
            dict(
                additional_dataset_parameters=dict(
                    intensity_postprocessing=train_modality,
                    image_base_folder=f'images_as_{dataset_name_train}',
                ),
                dataset_folder_subpath='multi_organ/bcv15_abdomen_4organs',
                has_validation_groundtruth=True,
                dataset_name_train=dataset_name_train,
                dataset_name_test='bcv15_abdomen_4organs',
                modality='ct',
            ),
            # --------------------------------------------------------
            # use source matched test domain data
            dict(
                additional_dataset_parameters=dict(
                    intensity_postprocessing=train_modality,
                    image_base_folder=f'images_as_{dataset_name_train}',
                ),
                dataset_folder_subpath='multi_organ/chaos19_mr_t2',
                has_validation_groundtruth=True,
                dataset_name_train=dataset_name_train,
                dataset_name_test='chaos19_mr_t2',
                modality='mr',
            ),
            # --------------------------------------------------------
            # use source matched test domain data
            dict(
                additional_dataset_parameters=dict(
                    intensity_postprocessing=train_modality,
                    image_base_folder=f'images_as_{dataset_name_train}',
                ),
                dataset_folder_subpath='multi_organ/amos22_4organs_ct',
                has_validation_groundtruth=True,
                dataset_name_train=dataset_name_train,
                dataset_name_test='amos22_4organs_ct',
                modality='ct',
            ),
            # --------------------------------------------------------

        ]

    # ------------------------------------------------------------

        for wrapper_dict in wrapper_dict_list:
            base_dataset_folder = os.path.join(base_dataset_folder_inference, wrapper_dict['dataset_folder_subpath'])

            run_test(
                cur_cv=cur_cv,
                setup=setup,
                base_dataset_folder=base_dataset_folder,
                base_output_folder=base_output_folder,
                num_labels=num_labels,
                image_size=image_size,
                image_extent=image_extent,
                is_inference=is_inference,
                params_dict=cur_params_dict,
                wrapper_dict=wrapper_dict,
            )

