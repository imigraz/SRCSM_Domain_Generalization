import os
import SimpleITK as sitk


def get_id_to_new_id_dict():
    id_dict = {
        'Prostate_000': 'prostate_A_000',
        'Prostate_001': 'prostate_A_001',
        'Prostate_002': 'prostate_A_002',
        'Prostate_003': 'prostate_A_003',
        'Prostate_004': 'prostate_A_004',
        'Prostate_005': 'prostate_A_005',
        'Prostate_006': 'prostate_A_006',
        'Prostate_007': 'prostate_A_007',
        'Prostate_008': 'prostate_A_008',
        'Prostate_009': 'prostate_A_009',
        'Prostate_010': 'prostate_A_010',
        'Prostate_011': 'prostate_A_011',
        'Prostate_012': 'prostate_A_012',
        'Prostate_013': 'prostate_A_013',
        'Prostate_014': 'prostate_A_014',
        'Prostate_015': 'prostate_A_015',
        'Prostate_016': 'prostate_A_016',
        'Prostate_017': 'prostate_A_017',
        'Prostate_018': 'prostate_A_018',
        'Prostate_019': 'prostate_A_019',
        'Prostate_020': 'prostate_A_020',
        'Prostate_021': 'prostate_A_021',
        'Prostate_022': 'prostate_A_022',
        'Prostate_023': 'prostate_A_023',
        'Prostate_024': 'prostate_A_024',
        'Prostate_025': 'prostate_A_025',
        'Prostate_026': 'prostate_A_026',
        'Prostate_027': 'prostate_A_027',
        'Prostate_028': 'prostate_A_028',
        'Prostate_029': 'prostate_A_029',
        'Prostate_030': 'prostate_B_030',
        'Prostate_031': 'prostate_B_031',
        'Prostate_032': 'prostate_B_032',
        'Prostate_033': 'prostate_B_033',
        'Prostate_034': 'prostate_B_034',
        'Prostate_035': 'prostate_B_035',
        'Prostate_036': 'prostate_B_036',
        'Prostate_037': 'prostate_B_037',
        'Prostate_038': 'prostate_B_038',
        'Prostate_039': 'prostate_B_039',
        'Prostate_040': 'prostate_B_040',
        'Prostate_041': 'prostate_B_041',
        'Prostate_042': 'prostate_B_042',
        'Prostate_043': 'prostate_B_043',
        'Prostate_044': 'prostate_B_044',
        'Prostate_045': 'prostate_B_045',
        'Prostate_046': 'prostate_B_046',
        'Prostate_047': 'prostate_B_047',
        'Prostate_048': 'prostate_B_048',
        'Prostate_049': 'prostate_B_049',
        'Prostate_050': 'prostate_B_050',
        'Prostate_051': 'prostate_B_051',
        'Prostate_052': 'prostate_B_052',
        'Prostate_053': 'prostate_B_053',
        'Prostate_054': 'prostate_B_054',
        'Prostate_055': 'prostate_B_055',
        'Prostate_056': 'prostate_B_056',
        'Prostate_057': 'prostate_B_057',
        'Prostate_058': 'prostate_B_058',
        'Prostate_059': 'prostate_B_059',
        'Prostate_060': 'prostate_C_060',
        'Prostate_061': 'prostate_C_061',
        'Prostate_062': 'prostate_C_062',
        'Prostate_063': 'prostate_C_063',
        'Prostate_064': 'prostate_C_064',
        'Prostate_065': 'prostate_C_065',
        'Prostate_066': 'prostate_C_066',
        'Prostate_067': 'prostate_C_067',
        'Prostate_068': 'prostate_C_068',
        'Prostate_069': 'prostate_C_069',
        'Prostate_070': 'prostate_C_070',
        'Prostate_071': 'prostate_C_071',
        'Prostate_072': 'prostate_C_072',
        'Prostate_073': 'prostate_C_073',
        'Prostate_074': 'prostate_C_074',
        'Prostate_075': 'prostate_C_075',
        'Prostate_076': 'prostate_C_076',
        'Prostate_077': 'prostate_C_077',
        'Prostate_078': 'prostate_C_078',
        'Prostate_079': 'prostate_D_079',
        'Prostate_080': 'prostate_D_080',
        'Prostate_081': 'prostate_D_081',
        'Prostate_082': 'prostate_D_082',
        'Prostate_083': 'prostate_D_083',
        'Prostate_084': 'prostate_D_084',
        'Prostate_085': 'prostate_D_085',
        'Prostate_086': 'prostate_D_086',
        'Prostate_087': 'prostate_D_087',
        'Prostate_088': 'prostate_D_088',
        'Prostate_089': 'prostate_D_089',
        'Prostate_090': 'prostate_D_090',
        'Prostate_091': 'prostate_D_091',
        'Prostate_092': 'prostate_E_092',
        'Prostate_093': 'prostate_E_093',
        'Prostate_094': 'prostate_E_094',
        'Prostate_095': 'prostate_E_095',
        'Prostate_096': 'prostate_E_096',
        'Prostate_097': 'prostate_E_097',
        'Prostate_098': 'prostate_E_098',
        'Prostate_099': 'prostate_E_099',
        'Prostate_100': 'prostate_E_100',
        'Prostate_101': 'prostate_E_101',
        'Prostate_102': 'prostate_E_102',
        'Prostate_103': 'prostate_E_103',
        'Prostate_104': 'prostate_F_104',
        'Prostate_105': 'prostate_F_105',
        'Prostate_106': 'prostate_F_106',
        'Prostate_107': 'prostate_F_107',
        'Prostate_108': 'prostate_F_108',
        'Prostate_109': 'prostate_F_109',
        'Prostate_110': 'prostate_F_110',
        'Prostate_111': 'prostate_F_111',
        'Prostate_112': 'prostate_F_112',
        'Prostate_113': 'prostate_F_113',
        'Prostate_114': 'prostate_F_114',
        'Prostate_115': 'prostate_F_115',
    }
    return id_dict


def main(in_base_path, out_base_path):
    in_file_ext = '.nii.gz'
    out_file_ext = '.nii.gz'

    out_image_dir = os.path.join(out_base_path, 'images')
    out_label_dir = os.path.join(out_base_path, 'labels')

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    id_dict = get_id_to_new_id_dict()

    in_image_dir = os.path.join(in_base_path, 'imagesTr')
    in_label_dir = os.path.join(in_base_path, 'labelsTr')

    in_image_list = sorted(os.listdir(in_image_dir))
    in_label_list = sorted(os.listdir(in_label_dir))

    old_ids = list(id_dict.keys())

    for old_id in old_ids:
        print(old_id)
        cur_in_image_filename = [x for x in in_image_list if old_id in x]
        cur_in_label_filename = [x for x in in_label_list if old_id in x]

        if len(cur_in_image_filename) != 1:
            print('Should be one, check code - exit')
            exit(42)
        if len(cur_in_label_filename) != 1:
            print('Should be one, check code - exit')
            exit(42)

        cur_in_image_filename = cur_in_image_filename[0]
        cur_in_label_filename = cur_in_label_filename[0]

        new_id = id_dict[old_id]
        cur_out_image_filename = f'{new_id}_image{out_file_ext}'
        cur_out_label_filename = f'{new_id}_label{out_file_ext}'

        image_sitk = sitk.ReadImage(os.path.join(in_image_dir, cur_in_image_filename))
        label_sitk = sitk.ReadImage(os.path.join(in_label_dir, cur_in_label_filename))

        orig_image_sitk = image_sitk
        image_np = sitk.GetArrayFromImage(orig_image_sitk)
        image_np = image_np * 1000
        image_sitk = sitk.GetImageFromArray(image_np)
        image_sitk.CopyInformation(orig_image_sitk)

        sitk.WriteImage(image_sitk, os.path.join(out_image_dir, cur_out_image_filename), useCompression=True)
        sitk.WriteImage(label_sitk, os.path.join(out_label_dir, cur_out_label_filename), useCompression=True)


if __name__ == "__main__":
    in_base_path = '/media0/franz/datasets/prostate/prostate_preprocessed_original'
    out_base_path = '/media0/franz/datasets/public_github_TEST/prostate/prostate_tr100'
    main(in_base_path, out_base_path)
