# Clasificación de implementos de seguridad con DenseNet121

Trabajo de Inteligencia Artificial II, UNAMBA. Jimmy Vargas Huarcaya.
Docente: Ing. Manuel Ibarra Cabrera.

DenseNet121 clasifica recortes en cuatro categorías: cabeza, casco, chaleco y trabajador. Utiliza la base preentrenada en ImageNet y un clasificador de cuatro salidas. Las anotaciones en formato YOLO permiten preparar los recortes. Los pesos de YOLO no intervienen en este modelo.

## Archivos

| Carpeta | Contenido |
|---|---|
| `scripts/` | Preparación de datos, entrenamiento, evaluación e inferencia |
| `demostraciones/` | Cuatro imágenes de prueba y sus predicciones |
| `outputs/resultados_finales/` | Configuración, métricas y gráficas del experimento |
| `outputs/datos/` | Manifiesto, distribución y muestras del dataset |
| `outputs/entorno/` | Registro del equipo y CUDA |
| `report/` | Informe final en Word y PDF |
| `slides/` | Presentación final |
| `references/` | Fuentes bibliográficas y procedencia de los datos |

El dataset, el entorno virtual y los pesos se conservan localmente y no se incluyen en el repositorio. El modelo está disponible en [Releases](https://github.com/JIMdvd/densenet-ppe-ia2/releases/tag/v1.0.0).

## Ejecución

Python 3.12. En PowerShell, desde la carpeta del proyecto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Descargar `densenet121_ppe.pt` desde Releases y colocarlo en `models/`. Ejecutar la demostración:

```powershell
.\.venv\Scripts\python.exe scripts/demo.py
```

Las predicciones aparecen en `demostraciones/resultados/`. La inferencia admite CPU o GPU. Para clasificar otra imagen recortada:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --image demostraciones/imagenes/helmet.jpg --output demostraciones/resultados/helmet.png
```

## Dataset y entrenamiento

[PPE Compliance Detection v5](https://universe.roboflow.com/yihong-chen/ppe-compliance-detection-iln63/dataset/5), Yihong Chen, licencia CC BY 4.0. Se utilizaron 2.000 fotografías originales. La limpieza de duplicados y el recorte de las anotaciones produjeron 8.493 muestras: 5.596 de entrenamiento, 1.816 de validación y 1.081 de prueba.

Para preparar una nueva copia del dataset y entrenar con GPU CUDA:

```powershell
.\.venv\Scripts\python.exe scripts/check_gpu.py
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --source "ruta_al_dataset_descomprimido"
.\.venv\Scripts\python.exe scripts/train.py --epochs 20 --batch-size 8 --image-size 128 --learning-rate 0.0005 --patience 4 --min-epochs 5
```

La preparación requiere una carpeta `data/classification/` nueva. El entrenamiento ajusta el clasificador final y mantiene congelada la base DenseNet121. Utiliza Adam, entropía cruzada y normalización ImageNet. Las versiones del entorno original están en `requirements-lock.txt`.

## Resultados

| Medida | Resultado |
|---|---|
| Épocas ejecutadas | 9 |
| Modelo seleccionado por validación | Época 5 |
| Exactitud de prueba | 97,69 % |
| F1 macro | 97,25 % |
| Aciertos | 1.056 de 1.081 |

Los resultados corresponden a clasificación de recortes del conjunto de prueba. El modelo no localiza objetos en escenas completas. La limpieza reduce duplicados entre conjuntos, aunque pueden permanecer escenas relacionadas. Las puntuaciones softmax no garantizan aciertos.
