# Semantic-aware Random Convolution and Source Matching for Domain Generalization in Medical Image Segmentation
This repository contains the official implementation of our paper:

***Semantic-aware Random Convolution and Source Matching for Domain Generalization in Medical Image Segmentation***
\
Published in [IEEE Access](https://ieeexplore.ieee.org/abstract/document/11493783) and also available on [arXiv](https://arxiv.org/abs/2512.01510).

If you use this code for your research, please cite our paper.

```
@article{thaler2026semantic,
  title={Semantic-Aware Random Convolution and Source Matching for Domain Generalization in Medical Image Segmentation}, 
  author={Thaler, Franz and Urschler, Martin and Koziński, Mateusz and Gsell, Matthias A. F. and Plank, Gernot and Štern, Darko},
  journal={IEEE Access}, 
  year={2026},
  volume={14},
  number={},
  pages={64338-64356},
  doi={10.1109/ACCESS.2026.3687116}
}
```
If you have questions about the code, write me a [mail](mailto:franz.thaler@medunigraz.at) or contact the corresponding author of the paper.



## Dependencies

To install dependencies and setup the environment, we recommend to use conda.
Use either [anaconda](https://www.anaconda.com/docs/getting-started/main) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main).
With conda installed and activated, the environment can be installed with the command below.
In the repository, the file `environment.yml` can be found in `bin/environment.yml`.

```
conda env create -f environment.yml
```


## Datasets

We used the following abdominal, cardiac and prostate datasets:



### Abdominal Datasets

- BCV (CT): [available here](https://www.synapse.org/Synapse:syn3193805/wiki/)
- CHAOS (MR): [available here](https://chaos.grand-challenge.org/)
- AMOS (CT & MR): [available here](https://amos22.grand-challenge.org/)



### Cardiac Datasets

- MMWHS (CT & MR): [available here](https://zmiclab.github.io/zxh/0/mmwhs/data.html)
- M&Ms-2 (Cine MR): [available here](https://www.ub.edu/mnms-2/)



### Prostate Datasets

- RUNMC/ISBI (A), BMC/ISBI1.5 (B), I2CVB (C), UCL (D), BIDMC (E) and HK (F): We used the prostate datasets as made [available here](https://github.com/yifangao112/DeSAM)



## Dataset Setup Folders

We prepared dataset-specific setup folders which contain dataset-specific information like the train-test split or center landmarks to determine the region of interest.
These folders also determine the dataset-specific folder names that several script build upon.
Therefore, we recommend to keep their names intact.
The setup folders are available in the repository in `bin/datasets`.
We suggest to move this `datasets` folder to a different directory on a drive with enough free storage to also allow storing the datasets within it.



## Data Preprocessing

We employ dataset specific preprocessing scripts to unify the structure and nomenclature of the datasets.
Preprocessing scripts are in `bin/preprocessing` and correspond to the respective datasets.
Note that for some datasets, we used multiple different evaluation protocols to allow a comparison to additional methods from related work. 
The variable `in_base_path` needs to be set to the base folder as acquired from the download link of the respective dataset.
The variable `out_base_path` needs to be set to the preferred output folder.
Some scripts require setting multiple output or intermediate paths.
For consistency with follow-up scripts, we recommend to keep the names of the folders that are appended to this path intact, e.g., `/heart/mmwhs17_ct/`.

Preprocessing scripts can then be called, e.g., as:


```
python 1_preprocess_bcv15_dataset.py
```


After preprocessing, all datasets follow the same general structure. For example:


    .                                       # The `base_dataset_folder` of the dataset
    ├── images                              # Image folder containing all training images
    │   ├── id001_image.nrrd            
    │   ├── ...                   
    │   └── id100_image.nrrd            
    ├── labels                              # Image folder containing all training masks
    │   ├── id001_label.nrrd            
    │   ├── ...                   
    │   └── id100_label.nrrd  
    └── setup                               # Setup folder as provided in this repository



## Source Matching

As explained in the paper, target domain data is matched to the average histogram of the source domain data.
For convenience, source matching is implemented as a preprocessing step for which scripts are located in `bin/source_matching`.
The implementation of source matching is separated into two scripts: one to compute the average source CDF of the respective datasets, and one to match target images to the respective average source CDF.

In both scripts, the variable `in_base_path` is expected to be set to the dataset folder that is named after the site, i.e., up to the folder `heart`, `multi_organ` or `prostate`.
To compute source matched images for multi_organ data the following scripts need to be called one after the other:

```
python 1_compute_average_cdf_multi_organ.py
```

```
python 2_source_matching_multi_organ.py
```

The source matched images will be computed and added to the respective dataset folders named, e.g., `images_as_mmwhs_ct`



## Train Models


To train a segmentation model, run the dataset-specific train script after defining the `base_dataset_folder` as well as the `base_output_folder`.
The `base_dataset_folder` needs to be set to the specific dataset.
For example, for MMWHS CT, the folder needs to be set to: `/PATH/TO/datasets/heart/mmwhs17_ct`.
The `base_output_folder` can be set to any directory, however, we recommend to follow our naming convention and set it to: `/PATH/TO/experiments/heart/mmwhs17_ct`.
A segmentation model can then be trained on MMWHS CT by calling:

```
python train_mmwhs17_ct.py
```


All hyperparameters in the scripts are set to their defaults as employed in our paper as proposed.
We recommend a GPU with 24 GB of memory for training and inference when using the suggested hyperparameters.


Note: To run the code, it might be necessary to set the following environment variable with the path adapted to the conda environment:

```
XLA_FLAGS="--xla_gpu_cuda_data_dir=/PATH/TO/CONDA/ENV/envs/srcsm_env"
```

Also, when not using an IDE, it might be necessary to set the `PYTHONPATH` environment variable to the base of this repository.



## Test Models on Other Datasets


After a model has been trained, it can be used to generate predictions on arbitrary target domains.
This is generally performed by running the dataset-specific test script after defining the path to the trained model as well as the path to the `base_dataset_folder` and to the `base_output_folder` (see `Train Models` on how to set the latter two).
To load a trained model and use it for inference, set the `load_model_base` path to the respective path of a trained model: for a model trained on MMWHS CT, set the variable in `test_mmwhs17_ct.py` to the path of a trained model up to the folder that encodes the date and time, e.g., `/PATH/TO/experiments/heart/mmwhs17_ct/unet/default/ct/cv1/setup/128/unet_RCNetPLabSm_it50000_lr5e-05/2026-04-13_11-04-10_496279`.

A model trained on, e.g., MMWHS CT, can then be evaluated on MMWHS MR by calling:
```
python test_mmwhs17_ct.py
```

The available dataset-specific test domains are already setup in the respective test script.






