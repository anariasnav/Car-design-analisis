"""
modCarpetas.py
=================

Script para la reestructuración de la base de datos cambiando su especificidad y conjuntos.

Autor: Andrés Arias Navarro
Fecha: 22/05/2023
"""

import os
import shutil
import errno

# Especificidad Marca
root = os.path.join('./BD-461/Especificidad MMA/test','')
for directory, subdir_list, file_list in os.walk(root):
    print('Directory:', directory)
    dir = directory.replace('./BD-461/Especificidad MMA/test','./BD-461/Especificidad M/test')
    dir2 = directory.replace('./BD-461/Especificidad MMA/test','./BD-461/Especificidad M/val')
    posicion = dir.find('_')
    posicion2 = dir2.find('_')
    ## Se añade para marca y modelo
    # posicion = dir.find('_',posicion+1)
    # posicion2 = dir2.find('_',posicion2+1)
    nueva = dir[:posicion]
    nueva_val = dir2[:posicion2]
    try:
        os.makedirs(nueva, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs(nueva_val, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  
    cont = 0
    for name in file_list:
        if (cont==0):
            cont+=1
            shutil.copy(os.path.join(directory,name),os.path.join(nueva_val,name))
        else:
            shutil.copy(os.path.join(directory,name),os.path.join(nueva,name))
        
        
root = os.path.join('./BD-461/Especificidad MMA/train','')
for directory, subdir_list, file_list in os.walk(root):
    print('Directory:', directory)
    dir = directory.replace('./BD-461/Especificidad MMA/train','./BD-461/Especificidad M/train')
    posicion = dir.find('_')
    #posicion = dir.find('_',posicion+1)
    nueva = dir[:posicion]
    try:
        os.makedirs(nueva, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise 
    
    for name in file_list:
        shutil.copy(os.path.join(directory,name),os.path.join(nueva,name))
        

# Modificación para crear el conjunto de datos para One-VS-All
#import os
#import shutil
#import errno
#
#root = os.path.join('./BD-461/Especificidad M/train','')
#for directory, subdir_list, file_list in os.walk(root):
#    print('Directory:', directory)
#    dir = directory.replace('./BD-461/Especificidad M/train','./BD-461/OVA/train')
#    posicion = dir.rfind('/')
#    nueva = dir[posicion+1:]
#    print(nueva)
#    if(nueva == 'Bmw'):
#        carpeta = dir[:posicion+1] + 'Bmw'
#    else:
#        carpeta =  dir[:posicion+1] + "otros"
#
#    print(carpeta)
#    try:
#        os.makedirs(carpeta, exist_ok=True)
#    except OSError as e:
#        if e.errno != errno.EEXIST:
#            raise
#
#    for name in file_list:
#        shutil.copy(os.path.join(directory,name),os.path.join(carpeta,name))
        