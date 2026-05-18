import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool


def process_image(cur_in_path_filename, output_image_path, remove_string=None, prefix=None, suffix=None):
    cur_in_path, cur_in_filename = os.path.split(cur_in_path_filename)
    cur_out_path_filename = os.path.join(output_image_path, cur_in_filename)

    print(cur_in_filename)

    image_sitk = sitk.ReadImage(cur_in_path_filename)
    image_np = sitk.GetArrayFromImage(image_sitk)
    image_np = np.flip(image_np, axis=1)  # flip axis 1
    out_image_sitk = sitk.GetImageFromArray(image_np)
    out_image_sitk.CopyInformation(image_sitk)

    cur_out_path, cur_out_file = os.path.split(cur_out_path_filename)

    if remove_string is not None:
        cur_out_file = cur_out_file.replace(remove_string, '')
    if cur_out_file.endswith('.nii.gz'):
        image_id = cur_out_file.replace('.nii.gz', '')
        file_ext = '.nii.gz'
    else:
        image_id, file_ext = os.path.splitext()

    out_image_file = image_id
    if prefix is not None:
        out_image_file = f'{prefix}{out_image_file}'
    if suffix is not None:
        out_image_file = f'{out_image_file}{suffix}'
    out_image_file = f'{out_image_file}{file_ext}'
    sitk.WriteImage(out_image_sitk, os.path.join(cur_out_path, out_image_file), useCompression=True)


def process_label(cur_in_path_filename, output_image_path, label_dict, remove_string=None, prefix=None, suffix=None):
    cur_in_path, cur_in_filename = os.path.split(cur_in_path_filename)
    cur_out_path_filename = os.path.join(output_image_path, cur_in_filename)

    print(cur_in_filename)

    image_sitk = sitk.ReadImage(cur_in_path_filename)
    image_np = sitk.GetArrayFromImage(image_sitk)

    # remap labels
    tmp_image_np = np.zeros_like(image_np)
    for k, v in label_dict.items():
        tmp_image_np = np.where(image_np == k, v, tmp_image_np)
    image_np = tmp_image_np

    image_np = np.flip(image_np, axis=1)  # flip axis 1
    out_image_sitk = sitk.GetImageFromArray(image_np)
    out_image_sitk.CopyInformation(image_sitk)

    cur_out_path, cur_out_file = os.path.split(cur_out_path_filename)

    if remove_string is not None:
        cur_out_file = cur_out_file.replace(remove_string, '')
    if cur_out_file.endswith('.nii.gz'):
        image_id = cur_out_file.replace('.nii.gz', '')
        file_ext = '.nii.gz'
    else:
        image_id, file_ext = os.path.splitext()

    out_image_file = image_id
    if prefix is not None:
        out_image_file = f'{prefix}{out_image_file}'
    if suffix is not None:
        out_image_file = f'{out_image_file}{suffix}'
    out_image_file = f'{out_image_file}{file_ext}'
    sitk.WriteImage(out_image_sitk, os.path.join(cur_out_path, out_image_file), useCompression=True)


def process_data(input_base_path, output_base_path):
    train_image_list = []
    train_label_list = []
    for root, dirs, files in os.walk(input_base_path):
        for file in files:
            # training images
            if file.endswith('.nii.gz') and 'Training' in root and 'img' in root:
                train_image_list.append(os.path.join(root, file))
            # training labels
            if file.endswith('.nii.gz') and 'Training' in root and 'label' in root:
                train_label_list.append(os.path.join(root, file))

    train_image_list = sorted(train_image_list)
    train_label_list = sorted(train_label_list)

    label_dict = {
        6: 1,
        2: 2,
        3: 3,
        1: 4,
    }

    output_image_path = os.path.join(output_base_path, 'images')
    os.makedirs(output_image_path, exist_ok=True)

    output_label_path = os.path.join(output_base_path, 'labels')
    os.makedirs(output_label_path, exist_ok=True)

    # multiprocessing - image
    with Pool(int(os.cpu_count() * 0.75)) as p:
        remove_string = 'img'
        prefix = 'id'
        suffix = '_image'
        p.starmap(process_image, [
            (
                cur_in_path_filename,
                output_image_path,
                remove_string,
                prefix,
                suffix,
            )
            for cur_in_path_filename in train_image_list])

    # multiprocessing - label
    with Pool(int(os.cpu_count() * 0.75)) as p:
        label_dict = label_dict
        remove_string = 'label'
        prefix = 'id'
        suffix = '_label'
        p.starmap(process_label, [
            (
                cur_in_path_filename,
                output_label_path,
                label_dict,
                remove_string,
                prefix,
                suffix,
            )
            for cur_in_path_filename in train_label_list])


def main():
    in_base_path = '/media0/franz/datasets/multi_organ/bcv15_abdomen_original/RawData'
    out_base_path = '/media0/franz/datasets/public_github_TEST/multi_organ/bcv15_abdomen_4organs'
    process_data(in_base_path, out_base_path)


if __name__ == '__main__':
    main()
