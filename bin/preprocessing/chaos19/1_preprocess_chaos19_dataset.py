import os
import SimpleITK as sitk
import numpy as np
import shutil


def convert_images(file_list, file_path, input_path, output_path, file_prefix=None, file_suffix=None, file_ext='.nii.gz', ref_image_sitk=None, label_dict=None):
    file_id = file_path.replace(input_path, '').split('/')[3].zfill(2)
    image_sitk = sitk.ReadImage([os.path.join(file_path, x) for x in file_list])
    if label_dict is not None:
        orig_image_np = sitk.GetArrayFromImage(image_sitk)
        image_np = np.zeros_like(orig_image_np)
        for k, v in label_dict.items():
            image_np = np.where(orig_image_np == k, v, image_np)
        print(f'{file_path} {file_id} - unique label values: {np.unique(image_np)}')
        image_sitk = sitk.GetImageFromArray(image_np)

    if ref_image_sitk is not None:
        image_sitk.CopyInformation(ref_image_sitk)

    cur_output_path = file_path.replace(input_path, output_path)
    os.makedirs(cur_output_path, exist_ok=True)
    out_file_id = file_id
    if file_prefix is not None:
        out_file_id = f'{file_prefix}{out_file_id}'
    if file_suffix is not None:
        out_file_id = f'{out_file_id}{file_suffix}'
    sitk.WriteImage(image_sitk, os.path.join(cur_output_path, f'{out_file_id}{file_ext}'), useCompression=True)
    return image_sitk


def walk_through_and_convert_images(input_path, output_path):
    for root, dirs, files in os.walk(input_path):

        # skip CT cases
        if 'CT' in [x for x in root.split('/')]:
            continue

        if 'CT' in [x for x in root.split('/')]:
            label_dict = {
                0: 0,
                255: 1,
            }

        if 'MR' in [x for x in root.split('/')]:
            label_dict = {
                0: 0,
                63: 1,
                126: 2,
                189: 3,
                252: 4,
            }

        image_sitk = None
        if 'DICOM_anon' in dirs:
            possible_subdirs = ['DICOM_anon', 'DICOM_anon/OutPhase', 'DICOM_anon/InPhase']
            for cur_possible_subdir in possible_subdirs:
                cur_file_path = os.path.join(root, cur_possible_subdir)
                if not os.path.isdir(cur_file_path):
                    continue
                dcm_file_list = os.listdir(cur_file_path)
                dcm_file_list = sorted([x for x in dcm_file_list if x.endswith('.dcm')])
                if len(dcm_file_list) > 0:
                    image_sitk = convert_images(dcm_file_list, cur_file_path, input_path, output_path, file_prefix='id', file_suffix='_image', file_ext='.nii.gz', ref_image_sitk=None, label_dict=None)

        if 'Ground' in dirs:
            cur_file_path = os.path.join(root, 'Ground')
            png_file_list = os.listdir(cur_file_path)
            png_file_list = sorted([x for x in png_file_list if x.endswith('.png')])
            label_sitk = convert_images(png_file_list, cur_file_path, input_path, output_path, file_prefix='id', file_suffix='_label', file_ext='.nii.gz', ref_image_sitk=image_sitk, label_dict=label_dict)


def flatten_folder_structure(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        file_list = sorted([x for x in files if x.endswith('.nii.gz')])

        if len(file_list) == 0:
            continue

        if 'CT' in [x for x in root.split('/')]:
            modality = 'ct'
        if 'MR' in [x for x in root.split('/')]:
            modality = 'mr'

        image_folder = f'{modality}'
        if 'Ground' in [x for x in root.split('/')]:
            image_folder = f'{image_folder}_labels'
        else:
            image_folder = f'{image_folder}_images'

        suffix = '_'.join([x.lower() for x in root.replace(input_path, '').split('/')[4:] if x != 'DICOM_anon' and x != 'Ground'])
        if len(suffix) > 0:
            image_folder = f'{image_folder}_{suffix}'
        if 'Test_Sets' in [x for x in root.split('/')]:
            image_folder = f'{image_folder}_test'

        cur_output_path = os.path.join(output_path, image_folder)
        os.makedirs(cur_output_path, exist_ok=True)
        for cur_file in file_list:
            shutil.copy2(os.path.join(root, cur_file), os.path.join(cur_output_path, cur_file))


def extract_chaos19_mr_t2(input_path, output_path):
    in_image_path = os.path.join(input_path, 'mr_images_t2spir')
    in_label_path = os.path.join(input_path, 'mr_labels_t2spir')
    out_image_path = os.path.join(output_path, 'images')
    out_label_path = os.path.join(output_path, 'labels')

    os.makedirs(out_image_path, exist_ok=True)
    image_list = os.listdir(in_image_path)
    for cur_file in image_list:
        print(cur_file)
        shutil.copy2(os.path.join(in_image_path, cur_file), os.path.join(out_image_path, cur_file))

    os.makedirs(out_label_path, exist_ok=True)
    label_list = os.listdir(in_label_path)
    for cur_file in label_list:
        print(cur_file)
        shutil.copy2(os.path.join(in_label_path, cur_file), os.path.join(out_label_path, cur_file))
    return


def main():
    in_base_path = '/media0/franz/datasets/multi_organ/chaos19_original/0_Original'
    out_base_path = '/media0/franz/datasets/public_github_TEST/multi_organ/chaos19_original/1_Converted'
    walk_through_and_convert_images(in_base_path, out_base_path)

    in_base_path = '/media0/franz/datasets/public_github_TEST/multi_organ/chaos19_original/1_Converted'
    out_base_path = '/media0/franz/datasets/public_github_TEST/multi_organ/chaos19_original/2_Flattened'
    flatten_folder_structure(in_base_path, out_base_path)

    in_base_path = '/media0/franz/datasets/public_github_TEST/multi_organ/chaos19_original/2_Flattened'
    out_base_path = '/media0/franz/datasets/public_github_TEST/multi_organ/chaos19_mr_t2'
    extract_chaos19_mr_t2(in_base_path, out_base_path)


if __name__ == '__main__':
    main()
