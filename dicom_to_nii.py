import os
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor

# DIRETÓRIOS
DIR_BASE = os.path.abspath('/mnt/d/ADNI/ADNI1')
DIR_DICOM = os.path.abspath(os.path.join(DIR_BASE, 'ADNI1_Screening', 'ADNI'))
DIR_RAW = os.path.join(DIR_BASE, 'ADNI_nii_raw')
DIR_OUTPUT = os.path.join(DIR_BASE, 'ADNI_nii_processed')

# FUNÇÕES
def arq_nii(name):
    return name + ".nii.gz"

def get_f_dir(directory):
    sub_item = os.listdir(directory)
    directory = os.path.abspath(os.path.join(directory, sub_item[0]))
    return directory

def load_dicom_series(input_folder):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(input_folder)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    return image

def save_as_nifti(image, output_file):
    sitk.WriteImage(image, output_file)

def convert_dicom_to_nifti(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Carrega a série DICOM
    image = load_dicom_series(input_folder)

    # Formar nome de saída
    output_name = arq_nii(os.path.basename(input_folder))

    # Nome do arquivo NIfTI de saída
    output_file = os.path.abspath(os.path.join(output_folder, output_name))

    # Salva no formato NIfTI
    save_as_nifti(image, output_file)

# CONVERSÃO
if __name__ == "__main__":
    input_folder = DIR_DICOM
    output_folder = DIR_RAW

    # Coletar todas as pastas DICOM
    dicom_folders = [os.path.join(DIR_DICOM, folder) for folder in os.listdir(DIR_DICOM)]

    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:  # Usando metade do número de CPUs
        # Mapeia a função convert_dicom_to_nifti para cada pasta DICOM
        executor.map(lambda folder: convert_dicom_to_nifti(
            get_f_dir(get_f_dir(os.path.join(folder, 'MP-RAGE'))), DIR_RAW), dicom_folders)


    print(f'Foram convertidas {len(dicom_folders)} imagens!')
