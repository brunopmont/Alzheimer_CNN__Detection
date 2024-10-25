import os
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor

# DIRETÓRIOS
DIR_BASE = os.path.abspath('/mnt/d/ADNI/ADNI2')
#DIR_DICOM = os.path.abspath(os.path.join(DIR_BASE, 'ADNI2_Screening', 'ADNI 2 Screening - New Pt', 'ADNI'))
DIR_DICOM = os.path.abspath(os.path.join(DIR_BASE, 'ADNI2_Screening'))
DIR_FILES = [folder for folder in os.listdir(DIR_DICOM)]
DIR_RAW = os.path.join(DIR_BASE, 'ADNI_nii_raw')

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

def reorient_image(image):
    # Reorienta a imagem para o sistema padrão RAS (Right, Anterior, Superior)
    return sitk.DICOMOrient(image, 'RAS')

def convert_dicom_to_nifti(input_folder, output_folder):

    # Formar nome de saída
    output_name = arq_nii(os.path.basename(input_folder))

    input_folder = get_f_dir(get_f_dir(os.path.join(input_folder, 'MPRAGE')))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Carrega a série DICOM
    image = load_dicom_series(input_folder)

    # Reorienta a imagem para o padrão RAS
    image = reorient_image(image)

    # Nome do arquivo NIfTI de saída
    output_file = os.path.abspath(os.path.join(output_folder, output_name))

    # Salva no formato NIfTI
    save_as_nifti(image, output_file)

    # Mensagem de conclusão
    print(f"Imagem {output_name} convertida com sucesso!")

# CONVERSÃO
if __name__ == "__main__":

    output_folder = DIR_RAW

    dir_files = DIR_FILES

    for names in dir_files:
        print(f"CONVERSÕES {names}")

        input_folder = os.path.abspath(os.path.join(DIR_BASE, 'ADNI2_Screening', names, 'ADNI'))

        # Coletar todas as pastas DICOM
        dicom_folders = [os.path.join(input_folder, folder) for folder in os.listdir(input_folder)]

        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            futures = {executor.submit(convert_dicom_to_nifti, folder, output_folder): folder for folder in dicom_folders}
            
            for future in futures:
                try:
                    future.result()  # Isso irá levantar qualquer exceção que ocorreu
                except Exception as e:
                    print(f"Erro ao processar {futures[future]}: {e}")

        print(f'Foram convertidas {len(dicom_folders)} imagens!')
