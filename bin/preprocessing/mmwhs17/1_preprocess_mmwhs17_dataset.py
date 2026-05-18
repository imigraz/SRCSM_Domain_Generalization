import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool


def process_one_image(out_base_path, cur_in_path_file, whspp_prefix, mmwhs_prefix, is_ct):
    print(cur_in_path_file)
    image_sitk = sitk.ReadImage(cur_in_path_file)

    if is_ct:
        clamp_val = -1023
        image_np = sitk.GetArrayFromImage(image_sitk)
        clamp_image_np = np.clip(image_np, clamp_val, None)
        clamp_image_sitk = sitk.GetImageFromArray(clamp_image_np)
        clamp_image_sitk.CopyInformation(image_sitk)
        image_sitk = clamp_image_sitk

    out_image_path = os.path.join(out_base_path, 'images')
    os.makedirs(out_image_path, exist_ok=True)
    cur_in_path, cur_in_file = os.path.split(cur_in_path_file)
    cur_out_file = cur_in_file.replace(whspp_prefix, mmwhs_prefix)
    sitk.WriteImage(image_sitk, os.path.join(out_image_path, cur_out_file), useCompression=True)


def process_one_label(out_base_path, cur_in_path_file, whspp_prefix, mmwhs_prefix):
    print(cur_in_path_file)
    label_dict = {
        500: 1,  # lv
        600: 2,  # rv
        420: 3,  # la
        550: 4,  # ra
        205: 5,  # myo
        820: 6,  # ao
        850: 7,  # pa
    }

    orig_image_sitk = sitk.ReadImage(cur_in_path_file)

    image_np = sitk.GetArrayFromImage(orig_image_sitk)
    for old_val, new_val in label_dict.items():
        image_np = np.where(image_np == old_val, new_val, image_np)

    image_sitk = sitk.GetImageFromArray(image_np)
    image_sitk.CopyInformation(orig_image_sitk)

    out_image_path = os.path.join(out_base_path, 'labels')
    os.makedirs(out_image_path, exist_ok=True)
    cur_in_path, cur_in_file = os.path.split(cur_in_path_file)
    cur_out_file = cur_in_file.replace(whspp_prefix, mmwhs_prefix)
    # image_sitk = sitk.DICOMOrient(image_sitk, 'lps')
    sitk.WriteImage(image_sitk, os.path.join(out_image_path, cur_out_file), useCompression=True)


def main(in_base_path, ct_out_base_path, mr_out_base_path):
    ct_train_image_list = []
    ct_train_label_list = []
    mr_train_image_list = []
    mr_train_label_list = []
    for root, dirs, files in os.walk(in_base_path):
        for file in files:

            # ct training images
            if file.endswith('.nii.gz') and 'CenterA' in root and 'image' in file:
                ct_train_image_list.append(os.path.join(root, file))
            # ct training labels
            if file.endswith('.nii.gz') and 'CenterA' in root and 'label' in file:
                ct_train_label_list.append(os.path.join(root, file))

            # mr training images
            if file.endswith('.nii.gz') and 'CenterCD' in root and 'image' in file:
                mr_train_image_list.append(os.path.join(root, file))
            # mr training labels
            if file.endswith('.nii.gz') and 'CenterCD' in root and 'label' in file:
                mr_train_label_list.append(os.path.join(root, file))


    ct_train_image_list = sorted(ct_train_image_list)
    ct_train_label_list = sorted(ct_train_label_list)
    mr_train_image_list = sorted(mr_train_image_list)
    mr_train_label_list = sorted(mr_train_label_list)


    # multiprocessing - ct images
    with Pool(int(os.cpu_count() * 0.75)) as p:
        p.starmap(process_one_image, [
            (
                ct_out_base_path,
                cur_file,
                'Case',
                'ct_train_',
                True,
            )
            for cur_file in ct_train_image_list])

    # multiprocessing - ct labels
    with Pool(int(os.cpu_count() * 0.75)) as p:
        p.starmap(process_one_label, [
            (
                ct_out_base_path,
                cur_file,
                'Case',
                'ct_train_',
            )
            for cur_file in ct_train_label_list])


    # multiprocessing - mr images
    with Pool(int(os.cpu_count() * 0.75)) as p:
        p.starmap(process_one_image, [
            (
                mr_out_base_path,
                cur_file,
                'Case3',
                'mr_train_1',
                False,
            )
            for cur_file in mr_train_image_list])

    # multiprocessing - mr labels
    with Pool(int(os.cpu_count() * 0.75)) as p:
        p.starmap(process_one_label, [
            (
                mr_out_base_path,
                cur_file,
                'Case3',
                'mr_train_1',
            )
            for cur_file in mr_train_label_list])



if __name__ == '__main__':

    # MMWHS is now part of CARE2024 WHS++
    # Thus, the original MMWHS dataset published in 2017 can be extracted from the WHS++ dataset
    in_base_path = '/media0/franz/datasets/heart/carewhspp24_orig/0_original/CARE2024_WHS++'
    out_base_path = '/media0/franz/datasets/public_github_TEST'

    ct_out_base_path = os.path.join(out_base_path, 'heart', 'mmwhs17_ct')
    mr_out_base_path = os.path.join(out_base_path, 'heart', 'mmwhs17_mr')

    main(in_base_path, ct_out_base_path, mr_out_base_path)
