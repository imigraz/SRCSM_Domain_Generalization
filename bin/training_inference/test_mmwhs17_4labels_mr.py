import os

from main import run_test


if __name__ == '__main__':

    # set according to dataset
    cur_cv = 0  # 0: uses test domain data
    setup = 'setup'

    base_dataset_folder_inference = '/PATH/TO/datasets'
    base_output_folder = '/PATH/TO/experiments/heart/mmwhs17_4labels_mr'

    dataset_name_train = 'mmwhs17_4labels_mr'
    train_modality = 'mr'
    num_labels = 5  # including background

    # image resampling parameters: lead to a spacing of 1.5, 1.5, 1.5
    image_size = [128, 128, 128]  # used in main paper
    # image_size = [96, 96, 96]  # can be used for faster templating with reduced memory footprint and computational cost
    image_extent = [192, 192, 192]

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
            #     dataset_folder_subpath='heart/mmwhs17_4labels_ct',
            #     has_validation_groundtruth=True,
            #     dataset_name_train=dataset_name_train,
            #     dataset_name_test='mmwhs17_4labels_ct',
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
                dataset_folder_subpath='heart/mmwhs17_4labels_ct',
                has_validation_groundtruth=True,
                dataset_name_train=dataset_name_train,
                dataset_name_test='mmwhs17_4labels_ct',
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

