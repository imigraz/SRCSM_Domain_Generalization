from main import run


if __name__ == '__main__':

    # set according to dataset
    modality = 'mr'
    cur_cv = 1
    setup = 'setup'

    base_dataset_folder = '/PATH/TO/datasets/heart/mmwhs17_4labels_mr'
    base_output_folder = '/PATH/TO/experiments/heart/mmwhs17_4labels_mr'

    num_labels = 5  # including background

    # image resampling parameters: lead to a spacing of 1.5, 1.5, 1.5
    image_size = [128, 128, 128]  # used in main paper
    # image_size = [96, 96, 96]  # can be used for faster templating with reduced memory footprint and computational cost
    image_extent = [192, 192, 192]

    # as used in main paper
    use_randnet = True
    use_randnet_per_label = True
    use_randnet_per_label_smoothing = True

    run(
        modality=modality,
        cur_cv=cur_cv,
        setup=setup,
        base_dataset_folder=base_dataset_folder,
        base_output_folder=base_output_folder,
        num_labels=num_labels,
        image_size=image_size,
        image_extent=image_extent,
        use_randnet=use_randnet,
        use_randnet_per_label=use_randnet_per_label,
        use_randnet_per_label_smoothing=use_randnet_per_label_smoothing,
    )

