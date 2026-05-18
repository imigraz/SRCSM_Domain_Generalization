import os
import SimpleITK as sitk
import numpy as np
import pandas as pd


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_image_np(image_np, ref_image_sitk, filename, output_path):
    image_sitk = sitk.GetImageFromArray(image_np)
    image_sitk.SetOrigin(ref_image_sitk.GetOrigin())
    image_sitk.SetSpacing(ref_image_sitk.GetSpacing())
    image_sitk.SetDirection(ref_image_sitk.GetDirection())
    create_dir(output_path)
    sitk.WriteImage(image_sitk, os.path.join(output_path, filename))


def get_file_ids(id_txt_file_path):
    df = pd.read_csv(id_txt_file_path, header=None)
    return list(np.squeeze(df.values))

