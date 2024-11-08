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
def winsorize_image(image_data, lower_percentile=1, upper_percentile=99):
    lower_bound = np.percentile(image_data, lower_percentile)
    upper_bound = np.percentile(image_data, upper_percentile)
    winsorized_data = np.clip(image_data, lower_bound, upper_bound)
    return winsorized_data

# Normalization -> valores de voxels entre 0 e 1
def normalize_image(image_data):
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_data = (image_data - min_val) / (max_val - min_val)
    return normalized_data

# Função para processar uma única imagem
def process_image(img_path, template, mask):
    try:
        logger.info(f"Inicio processamento: {img_path}")
        # Carrega a imagem
        image = ants.image_read(img_path)
        data = image.numpy()
        
        # Winsorizing
        data = winsorize_image(data)

        # Bias Field Correction
        image = ants.from_numpy(data, origin=image.origin, spacing=image.spacing, direction=image.direction)
        image = ants.n4_bias_field_correction(image, shrink_factor=2)

        # Reaplica winsorizing
        data = image.numpy()
        data = winsorize_image(data)
        image = ants.from_numpy(data, origin=image.origin, spacing=image.spacing, direction=image.direction)

        # Registro (Registration) com as transformações para a imagem e máscara, com intuito de melhorar o corte
        registration = ants.registration(fixed=template, moving=image, type_of_transform='Translation')
        warped_image = registration['warpedmovout']

        registration = ants.registration(fixed=template, moving=warped_image, type_of_transform='Rigid')
        warped_image = registration['warpedmovout']

        registration = ants.registration(fixed=template, moving=warped_image, type_of_transform='Affine')
        warped_image = registration['warpedmovout']

        registration = ants.registration(fixed=template, moving=warped_image, type_of_transform='SyN')
        warped_image = registration['warpedmovout']

        # Máscara do cérebro e extração
        brain_masked = ants.mask_image(warped_image, mask)

        # Normalização
        normalized_data = normalize_image(brain_masked.numpy())
        normalized_image = ants.from_numpy(normalized_data, origin=brain_masked.origin, spacing=brain_masked.spacing, direction=brain_masked.direction)

        logger.info(f"Imagem {img_path} processada.")
        return normalized_image
        
    except Exception as e:
        logger.error(f"Erro ao processar a imagem {img_path}: {e}")
        return None

# Função que processa e salva uma imagem
def process_and_save_image(img_path, template, mask, output_dir, subfolder):
    normalized_image = process_image(img_path, template, mask)
    if normalized_image is not None:
        output_path = os.path.join(output_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_path, os.path.basename(img_path))
        ants.image_write(normalized_image, output_path)
        logger.info(f"Imagem salva: {output_path}")
        # Liberar a memória manualmente
        normalized_image = None
        gc.collect()

# Início do processamento
if __name__ == "__main__":
    # DIRETÓRIOS
    DIR_RAW = ''
    DIR_OUTPUT = ''
    DIR_MASK = ''

    if len(sys.argv < 1):
        DIR_RAW = os.path.abspath('/mnt/e/ADNI_8k/1.7K - REPEAT')
        DIR_OUTPUT = os.path.abspath('/mnt/e/full_nii_adni')
        DIR_MASK = os.path.abspath('/mnt/e/ADNI/ADNI2/mni_icbm152_nlin_asym_09c')
    else:
        DIR_RAW = sys.argv[1]
        DIR_OUTPUT = sys.argv[2]
        DIR_MASK = sys.argv[3]

    template_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')
    mask_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')

    template = ants.image_read(template_path)
    mask = ants.image_read(mask_path)

    start_time = datetime.now()
    logger.info(f"Início do processamento em: {start_time}")

    for names in os.listdir(DIR_RAW):

        input_path = os.path.join(DIR_RAW, names)

        # Lista de caminhos para as imagens brutas
        already_processed = [file for file in os.listdir(DIR_OUTPUT)]
        image_paths = [os.path.join(input_path, file) for file in os.listdir(input_path) if file not in already_processed]

        # Função parcial para passar parâmetros fixos
        process_func = partial(process_and_save_image, template=template, mask=mask, output_dir=DIR_OUTPUT, subfolder=names)

        # Processamento e salvamento de cada imagem usando ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_func, img_path) for img_path in image_paths]
            
            for future in as_completed(futures):
                future.result()  # Pega o resultado para garantir que exceções sejam lançadas

    # Fim do processamento
    end_time = datetime.now()
    logger.info(f"Término do processamento em: {end_time}")
    logger.info(f"Duração total: {end_time - start_time}")