#!/usr/bin/env python3
import os
import numpy as np
import ants
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from concurrent.futures import as_completed
import gc
import sys

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# FUNÇÕES
# Winsorize -> reduz outliers, limitando os percentis inf e sup
def winsorize_image(image_data, lower_percentile=1, upper_percentile=99): #reduz valores extremos
    lower_bound = np.percentile(image_data, lower_percentile)
    upper_bound = np.percentile(image_data, upper_percentile)
    winsorized_data = np.clip(image_data, lower_bound, upper_bound)
    return winsorized_data

# Normalization -> valores de voxels entre 0 e 1
def normalize_image_min(image_data): 
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_data = (image_data - min_val) / (max_val - min_val)
    return normalized_data

def normalize_image_mean_std(image_data): #normaliza pela média e desvio padrão
    mean_val = np.mean(image_data)
    std_val = np.std(image_data)
    normalized_data = (image_data - mean_val) / std_val  # Normaliza para média 0 e desvio padrão 1
    return normalized_data

def clip_image(image_data, lower_percentile=2, upper_percentile=98):
    lower = np.percentile(image_data, lower_percentile)
    upper = np.percentile(image_data, upper_percentile)
    clipped_data = np.clip(image_data, lower, upper)
    return clipped_data

# Função para processar uma única imagem
def process_image(img_path, template, mask):
    try:
        logger.info(f"Inicio processamento: {os.path.basename(img_path)}")
        # Carrega a imagem
        image = ants.image_read(img_path)

        # Registro (Registration) com as transformações para a imagem e máscara, com intuito de melhorar o corte
        registration = ants.registration(fixed=template, moving=image, type_of_transform='Translation')
        warped_image = registration['warpedmovout']
        #logger.info(f"TRANSLATION")

        registration = ants.registration(fixed=template, moving=warped_image, type_of_transform='Rigid')
        warped_image = registration['warpedmovout']
        #logger.info(f"RIGID")

        registration = ants.registration(fixed=template, moving=warped_image, type_of_transform='Affine')
        warped_image = registration['warpedmovout']
        #logger.info(f"AFFINE")

        registration = ants.registration(fixed=template, moving=image, type_of_transform='SyN')
        warped_image = registration['warpedmovout']
        #logger.info(f"SYN")

        # Máscara do cérebro e extração
        brain_masked = ants.mask_image(warped_image, mask)
        data = brain_masked.numpy()

        # Winsorizing
        data = winsorize_image(data, lower_percentile=2, upper_percentile=99)
        #logger.info(f"WINSORIZE")

        # Bias Field Correction
        image = ants.from_numpy(data, origin=image.origin, spacing=image.spacing, direction=image.direction)
        image = ants.n4_bias_field_correction(image, shrink_factor=2)
        #logger.info(f"BIAS")

        # Reaplica winsorizing
        data = image.numpy()
        data = winsorize_image(data, lower_percentile=2, upper_percentile=99)
        image = ants.from_numpy(data, origin=image.origin, spacing=image.spacing, direction=image.direction)
        #logger.info(f"WINSORIZE AGAIN")

        # Normalização
        normalized_data = normalize_image_min(brain_masked.numpy())
        normalized_image = ants.from_numpy(normalized_data, origin=brain_masked.origin, spacing=brain_masked.spacing, direction=brain_masked.direction)

        #logger.info(f"Imagem {os.path.basename(img_path)} processada.")
        return normalized_image
        
    except Exception as e:
        logger.error(f"Erro ao processar a imagem {os.path.basename(img_path)}: {e}")
        return None

# Função que processa e salva uma imagem
def process_and_save_image(img_path, template, mask, output_dir):
    normalized_image = process_image(img_path, template, mask) #processa imagem
    if normalized_image is not None:
        os.makedirs(output_dir, exist_ok=True) #cria o diretório onde será salvo caso não exista
        output_path = os.path.join(output_dir, os.path.basename(img_path)) 
        ants.image_write(normalized_image, output_path)
        logger.info(f"Imagem salva: {os.path.basename(output_path)}")
        # Liberar a memória manualmente
        normalized_image = None
        gc.collect()

# Início do processamento
if __name__ == "__main__":
    # DIRETÓRIOS
    DIR_RAW = '' #diretório raiz de onde as imagens estão não processadas
    DIR_OUTPUT = '' #diretório onde as imagens serão processadas
    DIR_MASK = '' #diretório com as máscaras

    if len(sys.argv) > 2: #passar endereços pela linha de comando
        DIR_RAW = sys.argv[1]
        DIR_OUTPUT = sys.argv[2]
        DIR_MASK = sys.argv[3]
    else: #passar os endereços aqui mesmo
        DIR_RAW = os.path.abspath("/mnt/c/Users/Team Taiane/Desktop/ADNI/ADNI_extended/RAW/MP-RAGE")
        DIR_OUTPUT = os.path.abspath(os.path.join("/mnt/c/Users/Team Taiane/Desktop/ADNI/ADNI_extended/PROCESSED", os.path.basename(DIR_RAW)))
        os.makedirs(DIR_OUTPUT, exist_ok=True)
        DIR_MASK = os.path.abspath("/mnt/c/Users/Team Taiane/Desktop/ADNI/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c")
    
    template_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')
    mask_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')

    template = ants.image_read(template_path) 
    mask = ants.image_read(mask_path) 

    start_time = datetime.now()
    logger.info(f"INICIO DO PROCESSAMENTO")

    for names in os.listdir(DIR_RAW): #itera as subpastas dentre o diretóiro original

        '''
        se for só um diretório com todas as imagens, 
        pode tirar esse for e deixar input path e output path 
        só com DIR RAW e DIR OUTPUT e tirar os.makedirs
        '''

        input_path = os.path.join(DIR_RAW, names)
        output_path = os.path.join(DIR_OUTPUT, names)
        os.makedirs(output_path, exist_ok=True)

        # Caminhos das imagens
        already_processed = [file for file in os.listdir(output_path)] #checa o diretório de saída pra ver se alguma imagem já foi processada
        image_paths = [os.path.join(input_path, file) for file in os.listdir(input_path) if file not in already_processed] #carrega o endereço das imagens não processadas

        # Função parcial para passar parâmetros fixos
        process_func = partial(process_and_save_image, template=template, mask=mask, output_dir=output_path)

        # Processamento e salvamento de cada imagem usando ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=4) as executor: #max_workers define o número máximo de processos paralelos
            # dependendo do pc, é melhor fazer um por vez, pois paralelizar pode deixar cada processo mais demorado sem hardware que aguente
            futures = [executor.submit(process_func, img_path) for img_path in image_paths]
            
            for future in as_completed(futures):
                future.result()  # Pega o resultado para garantir que exceções sejam lançadas


    # Fim do processamento
    end_time = datetime.now()
    logger.info(f"FIM DO PROCESSAMENTO")
    logger.info(f"Duração total: {end_time - start_time}")