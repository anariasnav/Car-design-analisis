import sys
import subprocess
import pkg_resources
import torch
torch.cuda.is_available()

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import gc #Garbage collector
from tqdm import tqdm # Progress bar

# TAMAÑO DE LAS IMÁGENES
IMG_SIZE = (256, 256)

# DIMENSIÓN DE LA CAPA DE TEMPLATES DE LOS model_trainedS
TEMPLATE_SIZE = 256

# Se establece el número de épocas
EPOCHS = 25

# Se establece el tamaño de batch
BATCH_SIZE = 64

# Se establece el learning rate inicial
LEARNING_RATE = 0.0005

# Directorio que contiene la base de datos sobre la que vamos a realizar el entrenamiento
MAIN_DIR ="./BD-461/Especificidad MMA/"

# Numero de clases a diferenciar en el clasificador
CLASES = 461

# Se engloban las transformaciones necesarias para los conjuntos de train y test
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    # Se incluye data augmentation para prevenir el overfitting y poder conseguir unos pesos con un mayor accuracy en validación y test
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_dataset = torchvision.datasets.ImageFolder(
    root = MAIN_DIR + "train",
    transform = train_transform
    )

train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

val_dataset = torchvision.datasets.ImageFolder(
    root = MAIN_DIR + "val",
    transform = train_transform
    )

val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
ETIQUETAS = train_dataset.classes

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
img = np.swapaxes(img,0,2)
img = np.swapaxes(img, 0,1)
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {ETIQUETAS[label]}")
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device {device}.")

# Solución al BUG HTTP Error 403: rate liit exceeded when loading model
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# RESNET 50
#model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
#model.fc = nn.Linear(model.fc.in_features, CLASES)
# Creamos una nueva cabeza para el modelo
#head = nn.Sequential(
#    nn.Linear(model.fc.in_features, model.fc.in_features//2),
#    nn.ReLU(),
#    nn.Linear(model.fc.in_features//2, CLASES)
#)
#model.fc = head

# DENSENET 121
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features,CLASES)

model.eval() # esto se ha quitado sin probar
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)  #  <3>
loss_fn = nn.CrossEntropyLoss()  #  <4>


def validate(model, train_dataloader, val_dataloader, device):
    for name, loader in [("train", train_dataloader), ("val", val_dataloader)]:
        correct = 0
        total = 0
        
        model.eval() # Ponemos nuestrom model_trained en modo evaluación

        with torch.no_grad():  # <1>
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) # <2>
                total += labels.shape[0]  # <3>
                correct += int((predicted == labels).sum())  # <4>

        print("Accuracy in {}: {:.2f}".format(name , correct / total))
    return (correct/total)




import datetime

def training_loop(n_epochs, optimizer, model, loss_fn, train_dataloader):
    model = model.to(device=device)
    best_acuracy = float('-inf')
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        model.train() # Modo train
        
        for imgs, labels in train_dataloader:
            imgs = imgs.to(device=device)  # <1>
            labels = labels.to(device=device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 5 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_dataloader)))
            acuracy_val = validate(model, train_dataloader, val_dataloader, device)
            if(acuracy_val < best_acuracy):
                print("Entrenamiento detenido por early stopping, se estaba sobreentrenando el modelo")
                break
            else:
                best_acuracy = acuracy_val
        if(acuracy_val < best_acuracy):
            break

    return model


model_trained = training_loop(
    n_epochs = 100,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_dataloader = train_dataloader,
)
#Guardamos los pesos de la red para poder utilizarla sin reentrenarla
import os
model_path = "./models/"
if not os.path.exists(model_path):
    os.mkdir(model_path)

torch.save(model_trained.state_dict(), model_path + 'MMA.pt')

model

#Procedemos al testeo del modelo entrenado
test_dataset = torchvision.datasets.ImageFolder(
    root = MAIN_DIR + "test",
    transform=test_transform
    )

test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


#modelo = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
#modelo.fc = nn.Linear(modelo.fc.in_features, CLASES)

modelo = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
modelo.classifier = nn.Linear(modelo.classifier.in_features,CLASES)
# Creamos una nueva cabeza para el modelo

pesos = torch.load('./models/MMA.pt',map_location='cuda')
#pesos = torch.load('./models/resnet50.pt',map_location='cuda')
modelo.load_state_dict(pesos)
modelo.to(device)
modelo.eval()
#model_trained.eval()

correct = 0
total = 0
true_labels = []
predicted_labels = []

num_images_to_display = 10
subset_images = []
subset_labels_true = []
subset_labels_pred = []


with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = modelo(images)
        _, predicted = torch.max(outputs.data, 1)
        
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Se obtiene un subconjunto de imágenes para mostrar junto a sus etiquetas
        subset_images.extend(images[:num_images_to_display].cpu())
        subset_labels_true.extend(labels[:num_images_to_display].cpu().numpy())
        subset_labels_pred.extend(predicted[:num_images_to_display].cpu().numpy())

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

accuracy = 100 * correct / total
print('Precisión en el conjunto de prueba: {}%'.format(accuracy))

# Calcular métricas
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print('Precisión: {:.2f}%'.format(accuracy * 100))
print('Precisión promedio: {:.2f}%'.format(precision * 100))
print('Exhaustividad promedio: {:.2f}%'.format(recall * 100))
print('Puntuación F1 promedio: {:.2f}%'.format(f1 * 100))

# Generar matriz de confusión
confusion_mtx = confusion_matrix(true_labels, predicted_labels)
print('Matriz de confusión:')
print(confusion_mtx)

# Generar informe de clasificación
print('Informe de clasificación:')
print(classification_report(true_labels, predicted_labels))
subset_images = torch.stack(subset_images)
subset_labels_true = np.array(subset_labels_true)
subset_labels_pred = np.array(subset_labels_pred)
# Mostrar las imágenes junto con las etiquetas reales y predichas
fig, axs = plt.subplots(5, 2, figsize=(24, 12))

for i in range(num_images_to_display):
    image = subset_images[i]
    label_true = ETIQUETAS[subset_labels_true[i]]
    label_pred = ETIQUETAS[subset_labels_pred[i]]
    
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)  # Ajustar los valores de píxel al rango [0, 1]
    
    axs[i % 5, i // 5].imshow(image)
    axs[i % 5, i // 5].axis('off')
    axs[i % 5, i // 5].set_title(f'True: {label_true}, Pred: {label_pred}')

plt.tight_layout()
plt.show()
