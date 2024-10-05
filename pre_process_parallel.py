#!/usr/bin/env python3
import os
import numpy as np
import ants
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

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
        #logger.info(f"Iniciando o processamento da imagem: {img_path}")
        image = ants.image_read(img_path)
        data = image.numpy()
        
        # Winsorizing
        data = winsorize_image(data)
        #logger.info(f"Winsorizing aplicado à imagem: {img_path}")

        # Bias Field Correction
        image = ants.from_numpy(data, origin=image.origin, spacing=image.spacing, direction=image.direction)
        image = ants.n4_bias_field_correction(image, shrink_factor=2)
        #logger.info(f"Correção de campo de viés aplicada à imagem: {img_path}")

        # Reaplica winsorizing
        data = image.numpy()
        data = winsorize_image(data)
        image = ants.from_numpy(data, origin=image.origin, spacing=image.spacing, direction=image.direction)

        # Registro (Registration) com as transformações
        warped_image = ants.registration(fixed=template, moving=image, type_of_transform='Translation')['warpedmovout']
        warped_image = ants.registration(fixed=template, moving=warped_image, type_of_transform='Rigid')['warpedmovout']
        warped_image = ants.registration(fixed=template, moving=warped_image, type_of_transform='Affine')['warpedmovout']
        warped_image = ants.registration(fixed=template, moving=warped_image, type_of_transform='SyN')['warpedmovout']
        #logger.info(f"Registro completo para a imagem: {img_path}")

        # Máscara do cérebro e extração
        registration = ants.registration(fixed=warped_image, moving=template, type_of_transform='SyN')
        brain_mask = ants.apply_transforms(moving=mask, fixed=registration['warpedmovout'], transformlist=registration['fwdtransforms'], interpolator='nearestNeighbor')
        brain_mask_dilated = ants.morphology(brain_mask, radius=4, operation='dilate', mtype='binary')
        brain_masked = ants.mask_image(warped_image, brain_mask_dilated)

        # Normalização
        normalized_data = normalize_image(brain_masked.numpy())
        normalized_image = ants.from_numpy(normalized_data, origin=brain_masked.origin, spacing=brain_masked.spacing, direction=brain_masked.direction)

        logger.info(f"Imagem {img_path} processada com sucesso.")
        return normalized_image

    except Exception as e:
        logger.error(f"Erro ao processar a imagem {img_path}: {e}")
        return None
    
def process_image_wrapper(img_path):
    return process_image(img_path, template, mask)

# DIRETÓRIOS
DIR_BASE = os.path.abspath('/home/brunop/external/ADNI/ADNI1')
DIR_RAW = os.path.join(DIR_BASE, 'ADNI_nii_raw')
DIR_OUTPUT = os.path.join(DIR_BASE, 'ADNI_nii_processed')
DIR_MASK = os.path.join(DIR_BASE, 'mni_icbm152_nlin_asym_09c')

template_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')
mask_path = os.path.join(DIR_MASK, 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')

template = ants.image_read(template_path, reorient='IAL')
mask = ants.image_read(mask_path, reorient='IAL')

# Lista de caminhos para as imagens brutas
image_paths = [os.path.join(DIR_RAW, file) for file in os.listdir(DIR_RAW)]

# Início do processamento
start_time = datetime.now()
logger.info(f"Início do processamento em: {start_time}")

# Processamento paralelo
with ProcessPoolExecutor(max_workers=32) as executor:
    results = list(executor.map(process_image_wrapper, image_paths))

# Salvando as imagens normalizadas
for img, img_path in zip(results, image_paths):
    if img is not None:
        output_path = os.path.abspath(os.path.join(DIR_OUTPUT, os.path.basename(img_path)))
        ants.image_write(img, output_path)
        logger.info(f"Imagem salva: {output_path}")

# Fim do processamento
end_time = datetime.now()
logger.info(f"Término do processamento em: {end_time}")
logger.info(f"Duração total: {end_time - start_time}")

logger.info(f"\n\n{len(results)} imagens processadas com sucesso!")
