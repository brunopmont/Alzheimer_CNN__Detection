import os
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
import sys

# FUNÇÕES
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
    # Formar nome de saída pelo 'S_...'
    output_name = os.path.basename(input_folder) + '.nii.gz'

    input_folder = get_f_dir(get_f_dir(os.path.join(input_folder, 'MPRAGE')))

    # Formar nome de saída peçp 'I...'
    #output_name = os.path.basename(input_folder) + '.nii.gz'

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

    DIR_BASE = ''
    DIR_RAW = ''

    # DIRETÓRIOS
    if len(sys.argv < 1):
        DIR_BASE = os.path.abspath('/mnt/e/ADNI_8k/1.7K - REPEAT')
        DIR_RAW = os.path.join('/mnt/e/full_dataset', os.path.basename(DIR_BASE))
    else:
        DIR_BASE = sys.argv[1] #Endereço do diretório com as subpastas que contem as pastas dos arquivos dicom
        DIR_RAW = os.path.join(sys.argv[2], os.path.basename(DIR_BASE)) #endereço do diretório de Saída

    for names in DIR_BASE:
        print(f"CONVERSOES DA PASTA {names}")

        input_folder = os.path.abspath(os.path.join(DIR_BASE, names, 'ADNI'))

        output_folder = os.path.abspath(os.path.join(DIR_RAW, names))

        # Coletar todas as pastas DICOM
        dicom_folders = [os.path.join(input_folder, folder) for folder in os.listdir(input_folder)]

        with ProcessPoolExecutor(16) as executor:
            futures = {executor.submit(convert_dicom_to_nifti, folder, output_folder): folder for folder in dicom_folders}
            
            for future in futures:
                try:
                    future.result()  # Relata erros
                except Exception as e:
                    print(f"Erro ao processar {futures[future]}: {e}")

        print(f'Foram convertidas {len(dicom_folders)} imagens!')
