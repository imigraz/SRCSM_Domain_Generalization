import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


class DataProcessor:
    def __init__(self, input_path, pid):
        self.input_path = input_path
        self.pid = pid
        self.file_ext = '.nii.gz'
        self.sa_ed_filename = f'{self.pid}_SA_ED'
        self.sa_es_filename = f'{self.pid}_SA_ES'
        self.sa_cine_filename = f'{self.pid}_SA_CINE'
        self.label_suffix = '_gt'

        self.sa_ed_image_filename = f'{self.sa_ed_filename}{self.file_ext}'
        self.sa_ed_label_filename = f'{self.sa_ed_filename}{self.label_suffix}{self.file_ext}'
        self.sa_es_image_filename = f'{self.sa_es_filename}{self.file_ext}'
        self.sa_es_label_filename = f'{self.sa_es_filename}{self.label_suffix}{self.file_ext}'
        self.sa_cine_image_filename = f'{self.sa_cine_filename}{self.file_ext}'

    def copy_data_to_output_file(self, output_path, out_file_ext='.nii.gz'):
        try:
            sa_ed_image_sitk = sitk.ReadImage(os.path.join(self.input_path, self.sa_ed_image_filename))
            sa_ed_label_sitk = sitk.ReadImage(os.path.join(self.input_path, self.sa_ed_label_filename))
            sa_es_image_sitk = sitk.ReadImage(os.path.join(self.input_path, self.sa_es_image_filename))
            sa_es_label_sitk = sitk.ReadImage(os.path.join(self.input_path, self.sa_es_label_filename))
        except:
            print(f'Warning: Encountered an error when reading {self.pid}. PID is ignored.')
            return

        image_output_path = os.path.join(os.path.join(output_path, 'images'))
        label_output_path = os.path.join(os.path.join(output_path, 'labels'))
        os.makedirs(image_output_path, exist_ok=True)
        os.makedirs(label_output_path, exist_ok=True)

        # using image_filename to get rid of label suffix
        sitk.WriteImage(sa_ed_image_sitk, os.path.join(image_output_path, self.sa_ed_image_filename.replace(self.file_ext, f'_image{out_file_ext}')), useCompression=True)
        sitk.WriteImage(sa_ed_label_sitk, os.path.join(label_output_path, self.sa_ed_image_filename.replace(self.file_ext, f'_label{out_file_ext}')), useCompression=True)
        sitk.WriteImage(sa_es_image_sitk, os.path.join(image_output_path, self.sa_es_image_filename.replace(self.file_ext, f'_image{out_file_ext}')), useCompression=True)
        sitk.WriteImage(sa_es_label_sitk, os.path.join(label_output_path, self.sa_es_image_filename.replace(self.file_ext, f'_label{out_file_ext}')), useCompression=True)


def main(base_input_path, base_output_path):
    patient_id_list = sorted(os.listdir(base_input_path))

    for cur_pid in tqdm(patient_id_list, 'processing ...'):
        data_processor = DataProcessor(os.path.join(base_input_path, cur_pid), cur_pid)
        data_processor.copy_data_to_output_file(output_path=base_output_path, out_file_ext='.nii.gz')

    return


if __name__ == '__main__':

    # Note: some warnings will occur when preprocessing this dataset. the output__NEEDED? folders 'images' and 'labels' should both include 704 .nii.gz files

    in_base_path = '/media0/franz/datasets/heart/MnMs21/MnM2/dataset'
    out_base_path = '/media0/franz/datasets/public_github_TEST/heart/mnms21_sa_labeled'
    main(in_base_path, out_base_path)
