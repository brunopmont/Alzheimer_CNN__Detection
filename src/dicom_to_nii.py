import os
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
import sys
from datetime import datetime

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
    #output_name = os.path.basename(input_folder) + '.nii.gz'

    #input_folder = get_f_dir(os.path.join(input_folder, 'MPRAGE'))

    # Formar nome de saída pelo 'I...'
    output_name = os.path.basename(input_folder) + '.nii.gz'

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

    tot_images = 0

    DIR_BASE = ''
    DIR_RAW = ''

    # DIRETÓRIOS
    if len(sys.argv) > 1:
        DIR_BASE = sys.argv[1] #Endereço do diretório com as subpastas que contem as pastas dos arquivos dicom
        DIR_RAW = os.path.join(sys.argv[2], os.path.basename(DIR_BASE)) #endereço do diretório de Saída
    else:
        DIR_BASE = os.path.abspath("C:/Users/Team Taiane/Desktop/ADNI/DICOM/MPRAGE_SENSE2")
        DIR_RAW = os.path.join("C:/Users/Team Taiane/Desktop/ADNI/RAW", os.path.basename(DIR_BASE))

    os.makedirs(DIR_RAW, exist_ok=True)

    print(f"Convertendo imagens de:\n{DIR_BASE}\npara:\n{DIR_RAW}")

    start_time = datetime.now()
    print(f"Início do processamento em: {start_time}")

    for names in os.listdir(DIR_BASE):
        print(f"CONVERSOES DA PASTA {names}")

        input_folder = os.path.join(DIR_BASE, names, 'ADNI')

        output_folder = os.path.join(DIR_RAW, names)

        os.makedirs(output_folder, exist_ok=True)

        # Coletar todas as pastas DICOM
        already_converted = [file for file in os.listdir(output_folder)]
        dicom_folders = []

        for folder in os.listdir(input_folder):
            subdir = os.path.join(input_folder, folder, os.path.basename(DIR_BASE))
            for subsubdir in os.listdir(subdir):
                subsubdir = os.path.join(subdir, subsubdir)
                for item in os.listdir(subsubdir):
                    file = os.path.join(subsubdir, item)
                    if os.path.basename(file) + '.nii.gz' not in already_converted:
                        dicom_folders.append(file)


        with ProcessPoolExecutor(8) as executor:
            futures = {executor.submit(convert_dicom_to_nifti, folder, output_folder): folder for folder in dicom_folders}
            
            for future in futures:
                try:
                    future.result()  # Relata erros
                except Exception as e:
                    print(f"Erro ao processar {futures[future]}: {e}")

        tot_images += len(dicom_folders)

        print(f'\nForam convertidas {len(dicom_folders)} imagens!')

    # Fim do processamento
    print(f'\nForam convertidas {len(dicom_folders)} imagens!')
    end_time = datetime.now()
    print(f"Término do processamento em: {end_time}")
    print(f"Duração total: {end_time - start_time}")
