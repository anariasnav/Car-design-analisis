"""
PlagiarismDetector.py
=================

Script para detectar las similitudes y posibles casos de plagio entre marcas, nos devolverá información de similitudes de nuestra marca con las demás.

Parámetros: 
    - Marca: Representa la marca para la que se comprobarán su grado de similitud con las demás
    - Prob: Probabilidad mínima para considerarlo como similitud

Autor: Andrés Arias Navarro
Fecha: 06/07/2023
"""

import torch
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device {device}.")
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='Script para reducir el ruido de las imágenes generadas de forma automática con la GAN.')
parser.add_argument('--Marca', type=str, default = 'Audi')
parser.add_argument('--Prob', type=str, default='0.01')
args = parser.parse_args()

# Clase a investigar
MARCA = args.Marca

# Probabilidad mínima para considerarlo como similitud
PROB = args.Prob

# TAMAÑO DE LAS IMÁGENES
IMG_SIZE = (256, 256)

# Se establece el tamaño de batch
BATCH_SIZE = 256

# Directorio que contiene la base de datos sobre la que vamos a realizar el entrenamiento
MAIN_DIR ="../BD-461/Especificidad M/"

# Numero de clases a diferenciar en el clasificador
CLASES = 34


# Transformaciones a aplicar sobre el conjunto de datos
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Cargamos el conjunto de test
test_dataset = torchvision.datasets.ImageFolder(
    root = MAIN_DIR + "test",
    transform=test_transform
    )

test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
ETIQUETAS = test_dataset.classes


# Cargamos el modelo de clasificación entrenado
modelo = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
modelo.classifier = nn.Linear(modelo.classifier.in_features, CLASES)
pesos = torch.load("../models/DenseNetM.pt", map_location='cuda')
modelo.load_state_dict(pesos)
modelo.to(device)
modelo.eval()

# Inicialización
Ajenos_propios = [0] * CLASES
ProbabilidadesA = array_de_arrays = [[] for _ in range(CLASES)]
MaxA = [0] * CLASES
Propios_ajenos = [0] * CLASES
ProbabilidadesP = array_de_arrays = [[] for _ in range(CLASES)]
MaxP = [0] * CLASES

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Se obtienen para cada imagen las 5 clases con mayor probabilidad su probabilidad, etiqueta...
        logits = modelo(images)
        probs = F.softmax(logits, dim=1)
        probs5 = probs.topk(5)
        cont=0
        for image in images:
            label = labels[cont].detach().cpu().numpy()
            for p,c in zip(probs5[0][cont].detach().cpu().numpy(), probs5[1][cont].detach().cpu().numpy()):
                if(ETIQUETAS[c]==MARCA and ETIQUETAS[label]!=MARCA and p>PROB):
                    Ajenos_propios[label] += 1
                    ProbabilidadesA[label].append(p)
                    if(p>MaxA[label]):
                        MaxA[label] = p
                elif(ETIQUETAS[label]==MARCA and ETIQUETAS[c]!=MARCA and p>PROB):
                    Propios_ajenos[c] +=1
                    ProbabilidadesP[c].append(p)
                    if(p>MaxP[c]):
                        MaxP[c] = p
            cont+=1
        # Liberar la memoria en la GPU
        del images, labels, logits, probs, probs5
        torch.cuda.empty_cache()

ProbAjenasMedias = [np.mean(prob) for prob in ProbabilidadesA]
ProbPropMedias = [np.mean(prob) for prob in ProbabilidadesP]

print("|   Marca   |   Instancias ajenas clasificadas   |   Probabilidad media de pertenecencia   |   Instancias propias clasificadas   |   Probabilidad media de pertenencia   |")
print("|           |            como propias            |      a nuestra clase siendo ajena       |             como ajenas             |               a otra clase            |")
print("|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|")
for i in range(CLASES):
    print(f"|   {ETIQUETAS[i]}         {Ajenos_propios[i]}                     {ProbAjenasMedias[i]}           {Propios_ajenos[i]}          {ProbPropMedias[i]}     {MaxA[i]}   {MaxP[i]}")

print("|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|")