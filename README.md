# Análisis del lenguaje de diseño de
vehículos mediante Deep Learning

## Descripción
TFG que busca evaluar y proponer diversas aplicaciones de algoritmos de IA al diseño automotriz.

## Estructuración
En este repositorio se encuentran los distintos códigos implementados para este proyecto, así como la memoria del mismo donde se explica el proceso llevado a cabo, los resultados obtenidos y las diversas aplicaciones de los algoritmos que se plantean.

En primera instancia encontramos la memoria del proyecto en el archivo _Memoria.pdf_. Además encontramos diferentes archivos correspondientes a los distintos algoritmos y enfoques evaluados.

* Los archivos _'Clasificador-M', 'Clasificador-MM' y 'Clasificador-MMA'_ llevan a cabo el preprocesado de datos, entrenamiento de las arquitecturas con los distintos datasets y testeo de los modelos entrenados para las diferentes especificidades.
* _'Lime-explainer'_ y _'SHAP'_ implementan las 2 técnicas de explicabilidad de modelos utilizadas para las distintas funciones de análisis expuestas en la memoria.
* _'GAN'_ implementa el proceso de entrenamiento y generación automática de imágenes haciendo uso de la arquitectura DCGAN.
* _'TEST'_ cuaderno de jupyter notebook para el etiquetado de instancias utilizando los modelos de clasificación entrenados. En concreto se utiliza para la clasificación de las imágenes generadas por la GAN.

También encontramos 3 subdirectorios:

### Auxiliary scripts
Directorio que contiene los scripts auxiliares implementados durante el desarrollo de este proyecto para manejar los resultados obtenidos de los modelos de entrenamiento. 
* _'modCarpetas'_ - Modifica la estructura de la base de datos para obtener diferentes especificidades en el conjunto de datos.
* _'Denoising'_ - Reduce el ruido de las imágenes generadas por la GAN
* _'Split'_ - Divide las cuadrículas de imágenes generadas por la GAN en imágenes individuales.
* _'PlagiarismDetector'_ - Script para la obtención de información de similitud entre múltiples marcas haciendo uso del clasificador de marca entrenado.

__Aviso__: La base de datos está disponible de forma temporal para la evaluación de este TFG.

### Models.zip
Contiene los pesos de los distintos modelos entrenados durante el desarrollo del presente proyecto. Dichos pesos se almacenan en un archivo formato `.pd` que nos permitirán cargar las redes entrenadas para posteriores testeos o aplicaciones de las mismas sin que se tengan que volver a entrenar. Debido al elevado tamaño de los modelos almacenados no se almacena en GitHub pero pueden descargarse en el siguiente enlace: [https://drive.google.com/file/d/1lOFsEJcboSXecwK-m1VVGVLEdcvu1Uo5/view?usp=sharing](https://drive.google.com/file/d/1lOFsEJcboSXecwK-m1VVGVLEdcvu1Uo5/view?usp=sharing) . Para realizar pruebas con los modelos ya entrenados será necesario descomprimirlo.

### Generated images
Contiene las imágenes generadas por las distintas arquitecturas evaluadas GAN y Stylegan.

### Indiv
Contiene las imágenes individuales para clasificar las  instancias sintéticas generadas por la GAN.

## Tecnologías usadas

Para el correcto desarrollo de este proyecto se han usado los siguientes frameworks, lenguajes, bibliotecas, etc:

- [Anaconda](https://www.anaconda.com/)
- [Python](https://www.python.org/)
- [Pytorch](https://pytorch.org/)
- [SHAP](https://shap.readthedocs.io/en/latest/index.html)
- [LIME](https://github.com/marcotcr/lime)

El hardware utilizado ha sido la GPU [NVIDIA V100](https://www.nvidia.com/en-us/data-center/v100/)
