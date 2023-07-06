import os
import shutil
import errno
import cv2
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--inDir', type=str, default = None)
parser.add_argument('--out', type=str, default='./Denoised Images/')
args = parser.parse_args()
DIR_in =  args.inDir
DIR = args.out

root = os.path.join(DIR_in,'')
for directory, subdir_list, file_list in os.walk(root):
    print('Directory:', directory)

    try:
        os.makedirs(DIR, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    for name in file_list:
        # Cargar la imagen generada
        generated_image = cv2.imread(directory + name)

        # Aplicar el filtrado bilateral
        denoised_image = cv2.bilateralFilter(generated_image, 9, 75, 75)

        # Guardar la imagen denoised
        cv2.imwrite(DIR + name, denoised_image)
