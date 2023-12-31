"""
Denoising.py
=================

Script para reducir el ruido de las imágenes generadas de forma automática con la GAN.

Parámetros: 
    - inDir: Representa el directorio del que se leen las imágenes a limpiar
    - out:   Directorio donde se almacenan las imágenes sin ruido. Su formato ha de ser ./nombre/

Autor: Andrés Arias Navarro
Fecha: 31/05/2023
"""

import os
import errno
import cv2
import argparse

parser = argparse.ArgumentParser(description='Script para reducir el ruido de las imágenes generadas de forma automática con la GAN.')
parser.add_argument('--inDir', type=str, default = None)
parser.add_argument('--out', type=str, default='../Denoised Images/')
args = parser.parse_args()
DIR_in =  args.inDir
DIR = args.out

try:
    os.makedirs(DIR, exist_ok=True)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

root = os.path.join(DIR_in,'')
for directory, subdir_list, file_list in os.walk(root):
    print('Directory:', directory)

    for name in file_list:
        # Cargar la imagen generada
        generated_image = cv2.imread(directory + name)

        # Aplicar el filtrado bilateral
        denoised_image = cv2.bilateralFilter(generated_image, 9, 75, 75)

        # Guardar la imagen denoised
        cv2.imwrite(DIR + name, denoised_image)
