import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool


def process_one(
        image_filename,
        label_filename,
        input_path,
        ct_output_path,
        mr_output_path,
        input_ext,
        output_ext,
        amos22_label_order_dict,
        output_label_order_dict,
        image_folder,
        label_folder,
        out_image_folder='images',
        out_label_folder='labels',
):
    cur_id = image_filename.replace(input_ext, '')
    cur_id_int = int(''.join([x for x in cur_id if x.isdigit()]))

    print(cur_id)

    # ID number determines if CT or MR
    if cur_id_int <= 500:
        cur_output_path = ct_output_path
    else:
        cur_output_path = mr_output_path

    cur_image_sitk = sitk.ReadImage(os.path.join(input_path, image_folder, image_filename))
    cur_label_sitk = sitk.ReadImage(os.path.join(input_path, label_folder, label_filename))

    cur_label_np = sitk.GetArrayFromImage(cur_label_sitk)

    # check label availability
    amos22_labels = list(amos22_label_order_dict.values())
    available_labels = np.unique(cur_label_np)

    available_label_keys = list([k for k, v in amos22_label_order_dict.items() if v in available_labels])
    # if not np.all([x in available_labels for x in amos22_labels]):
    if not np.all([x in available_label_keys for x in output_label_order_dict.keys()]):
        print('Warning: not all labels available (ignored)!', os.path.join(input_path, label_folder, label_filename))
        return

    # convert label file
    out_label_np = np.zeros_like(cur_label_np)
    for key, new_val in output_label_order_dict.items():
        orig_val = amos22_label_order_dict[key]
        out_label_np = np.where(cur_label_np == orig_val, new_val, out_label_np)

    # write label file
    out_label_sitk = sitk.GetImageFromArray(out_label_np)
    out_label_sitk.SetOrigin(cur_label_sitk.GetOrigin())
    out_label_sitk.SetSpacing(cur_label_sitk.GetSpacing())
    out_label_sitk.SetDirection(cur_label_sitk.GetDirection())
    for key in cur_label_sitk.GetMetaDataKeys():
        out_label_sitk.SetMetaData(key, cur_label_sitk.GetMetaData(key))

    cur_label_output_path = os.path.join(cur_output_path, out_label_folder)
    os.makedirs(cur_label_output_path, exist_ok=True)
    sitk.WriteImage(out_label_sitk, os.path.join(cur_label_output_path, f'{cur_id}_label{output_ext}'), useCompression=True)

    # write image file
    cur_image_output_path = os.path.join(cur_output_path, out_image_folder)
    os.makedirs(cur_image_output_path, exist_ok=True)
    sitk.WriteImage(cur_image_sitk, os.path.join(cur_image_output_path, f'{cur_id}_image{output_ext}'), useCompression=True)


def main(input_path, ct_output_path, mr_output_path):

    input_ext = '.nii.gz'
    output_ext = '.nii.gz'

    amos22_label_order_dict = {
        "background": 0,
        "spleen": 1,
        "right kidney": 2,
        "left kidney": 3,
        "gall bladder": 4,
        "esophagus": 5,
        "liver": 6,
        "stomach": 7,
        "aorta": 8,
        "postcava": 9,
        "pancreas": 10,
        "right adrenal gland": 11,
        "left adrenal gland": 12,
        "duodenum": 13,
        "bladder": 14,
        "prostate/uterus": 15
    }

    output_label_order_dict = {
        "background": 0,
        "spleen": 4,
        "right kidney": 2,
        "left kidney": 3,
        "liver": 1,
    }

    folder_pair_list = [
        ['imagesTr', 'labelsTr'],
        ['imagesVa', 'labelsVa'],
    ]

    for image_folder, label_folder in folder_pair_list:
        image_list = sorted(os.listdir(os.path.join(input_path, image_folder)))
        label_list = sorted(os.listdir(os.path.join(input_path, label_folder)))

        new_image_list = []
        new_label_list = []
        for image_filename in image_list:
            if not image_filename in label_list:
                print(f'Warning: cur file not found in label folder (ignored): {image_filename} not in {os.path.join(input_path, label_folder)}')
                continue
            new_image_list.append(image_filename)
            new_label_list.append(image_filename)

        # multiprocessing
        with Pool(int(os.cpu_count() * 0.75)) as p:
            p.starmap(process_one, [
                (
                    image_filename,
                    label_filename,
                    input_path,
                    ct_output_path,
                    mr_output_path,
                    input_ext,
                    output_ext,
                    amos22_label_order_dict,
                    output_label_order_dict,
                    image_folder,
                    label_folder,
                 )
                for image_filename, label_filename in zip(new_image_list, new_label_list)])

    return


if __name__ == '__main__':

    in_base_path = '/media0/franz/datasets/multi_organ/amos22/0_Original_amos22'
    ct_out_base_path = '/media0/franz/datasets/public_github_TEST/multi_organ/amos22_4organs_ct'
    mr_out_base_path = '/media0/franz/datasets/public_github_TEST/multi_organ/amos22_4organs_mr'
    main(in_base_path, ct_out_base_path, mr_out_base_path)
