"""
Split.py
=================

Script para la división de las cuadrículas de imágenes generadas de forma automática con la GAN a imágenes inviduales.


Parámetros: 
    - inDir: Representa el directorio del que se leen las imágenes a dividir
    - out:   Directorio donde se almacenan las imágenes individuales. Su formato ha de ser ./nombre/


Autor: Andrés Arias Navarro
Fecha: 01/07/2023
"""

from PIL import Image
import os
import errno
import argparse

parser = argparse.ArgumentParser(description='Script para la división de las cuadrículas de imágenes generadas de forma automática con la GAN a imágenes inviduales.')
parser.add_argument('--inDir', type=str, default = None)
parser.add_argument('--out', type=str, default='../Split Images/')
args = parser.parse_args()
In =  args.inDir
Out = args.out


try:
    os.makedirs(Out, exist_ok=True)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

cont = -1
for directory, subdir_list, file_list in os.walk(In):
    for name in file_list:
        cont+=1

        # Abrir la imagen grande
        imagen_original = Image.open(In + name)

        # Obtener el tamaño de la imagen grande
        ancho, alto = imagen_original.size

        # Definir el tamaño de cada imagen de la cuadrícula
        ancho2 = ancho // 5  # Dividir el ancho por el número de columnas de la cuadrícula
        alto2 = alto // 5

        # Recorrer la cuadrícula y extraer cada imagen
        for i in range(5):  # Filas de la cuadrícula
            for j in range(5):  # Columnas de la cuadrícula
                # Calcular las coordenadas de la región de interés
                left = j * ancho2
                top = i * alto2
                right = left + ancho2
                bottom = top + alto2

                # Extraer la imagen de la región de interés
                imagen_extraida = imagen_original.crop((left, top, right, bottom))

                # Guardar la imagen extraída
                nombre_archivo = os.path.join(Out,f'imagen_extraida_{cont*5+i}_{cont*5+j}.jpg')
                imagen_extraida.save(nombre_archivo)
