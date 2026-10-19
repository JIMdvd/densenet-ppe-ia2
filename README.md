# Clasificacion de regiones de seguridad en obras con DenseNet121

Proyecto del curso Inteligencia Artificial II. La tarea consiste en clasificar recortes de cuatro categorias: cabeza (Head), casco (Helmet), chaleco (Vest) y trabajador (Worker).

Repositorio: https://github.com/JIMdvd/densenet-ppe-ia2

## Uso en otra computadora

El repositorio conserva el codigo, el historial Git, las metricas y las demostraciones. No incluye el entorno virtual ni el dataset completo. Para ejecutar los ejemplos, instalar Python 3.12 y, desde una terminal PowerShell:

```powershell
git clone https://github.com/JIMdvd/densenet-ppe-ia2.git
cd densenet-ppe-ia2
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Descargar `densenet121_ppe.pt` desde [Releases](https://github.com/JIMdvd/densenet-ppe-ia2/releases) y colocarlo en `models/` (crear la carpeta si no existe). Ejecutar `.\.venv\Scripts\python.exe scripts\demo.py`. La prediccion usa CPU si CUDA no esta disponible; el entrenamiento exige GPU CUDA. `requirements-lock.txt` registra las versiones exactas del entorno original.

Para reentrenar, descargar el dataset de la fuente indicada abajo y ejecutar `scripts/prepare_dataset.py --source "ruta_al_dataset_descomprimido"` con el Python del entorno, antes de entrenar. No regenerar datos sobre una carpeta `data/classification/` existente. Los resultados finales conservados corresponden al experimento documentado, no a cualquier nueva ejecucion.

## Organizacion

Empieza por `EMPIEZA_AQUI.md`: mapa de carpetas y demostracion en un comando.

- `data/raw/`: copia de las 2000 imagenes originales y sus cajas etiquetadas.
- `data/classification/`: recortes organizados por conjunto y categoria para DenseNet.
- `scripts/`: preparacion de datos y comprobacion de GPU.
- `outputs/resultados_finales/`: graficas, metricas, configuracion y checkpoints definitivos, separados en subcarpetas.
- `outputs/datos/`: conteos, manifiesto de recortes y muestras del dataset.
- `outputs/entorno/`: comprobacion de CUDA.
- `outputs/pruebas_preliminares/`: ensayo anterior; no usar como resultado final.
- `outputs/entrenamientos/`: destino de nuevas ejecuciones, si se realizan.
- `demostraciones/imagenes/`: cuatro recortes de ejemplo con nombres simples.
- `demostraciones/resultados/`: predicciones y graficas de demostracion, separadas de las metricas del experimento.
- `references/yolo/`: configuracion y resultados del entrenamiento YOLO anterior, como antecedente.
- `models/`: destino previsto para el modelo DenseNet entrenado.
- `report/` y `slides/`: espacios para informe y exposicion.
- `.venv/`: entorno de Python de este proyecto.

## Datos y alcance

Origen: PPE Compliance Detection, version 5, Roboflow, licencia CC BY 4.0.
https://universe.roboflow.com/yihong-chen/ppe-compliance-detection-iln63/dataset/5

Separacion original: 1400 imagenes train, 400 valid y 200 test. Se conserva esta separacion antes de generar recortes. Se agrupan duplicados exactos (SHA256) y casi identicos (distancia dHash <=6 y diferencia RGB media <=3 sobre 255 a 64x64). Cada grupo se conserva en un solo conjunto, con prioridad test, valid, train. El manifiesto conserva la trazabilidad y el resumen registra exclusiones. Este filtro conservador no garantiza separar todas las escenas relacionadas o fotogramas del mismo video.

Las etiquetas YOLO se utilizan como anotaciones de ubicacion para recortar objetos, sin ejecutar YOLO. Un recorte de Worker puede incluir casco o chaleco, pero su etiqueta indica el tipo de region, Worker. Este clasificador no determina por si solo si una persona cumple todas las normas de seguridad. Los resultados YOLO anteriores corresponden a deteccion y no son directamente comparables con la exactitud de clasificacion de DenseNet.

## Entorno

Python 3.12, PyTorch 2.7.1 y torchvision 0.22.1 con CUDA 11.8. El controlador NVIDIA existente es compatible; no se modifica el controlador ni se instala un CUDA Toolkit global.

Desde una terminal PowerShell en esta carpeta:

```powershell
.\.venv\Scripts\python.exe scripts\check_gpu.py
```

La comprobacion debe indicar `cuda_available: true` y `cuda_operation_ok: true` antes de entrenar.

## Entrenamiento

```powershell
.\.venv\Scripts\python.exe scripts\train.py --epochs 5 --batch-size 8 --image-size 128
```

Se entrenan los pesos del clasificador final y se conserva congelada la base DenseNet121 preentrenada en ImageNet, incluyendo las estadisticas de BatchNorm. La funcion de perdida es entropia cruzada para cuatro clases. El redimensionamiento a 128 por 128 y la normalizacion ImageNet se aplican a todos los conjuntos. La configuracion inicial no incluye aumentos de datos.

Se selecciona el modelo con menor perdida en validacion y se evalua en test al finalizar. Cada ejecucion crea una carpeta de resultados con configuracion, historial, curvas, matriz de confusion, predicciones y metricas por clase. El tiempo restante se estima despues de la primera epoca. La evaluacion por recorte no equivale a una evaluacion por escena y las escenas relacionadas pueden introducir dependencia aun tras quitar duplicados exactos.

Configuracion definitiva, con un maximo de 20 epocas y parada tras cuatro epocas sin reducir la perdida de validacion al menos 0.001:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --epochs 20 --batch-size 8 --image-size 128 --learning-rate 0.0005 --patience 4 --min-epochs 5
```

Esta ejecucion empieza desde los pesos ImageNet y un clasificador nuevo, no desde el ensayo preliminar. Se guardan una copia del manifiesto y del resumen del dataset junto a cada ejecucion. El ensayo de cinco epocas del 19 de octubre a las 10:21 es preliminar: se detectaron duplicados visuales adicionales y sus metricas no deben presentarse como resultados finales.

## Demostracion con el modelo guardado

Despues de entrenar, se puede clasificar un recorte sin volver a entrenar:

```powershell
.\.venv\Scripts\python.exe scripts\demo.py
```

El script recupera las clases y el tamano de entrada del checkpoint. Funciona en GPU o con `--device cpu`. Las puntuaciones softmax no son una garantia de acierto. La entrada debe ser una region recortada: este modelo no genera cajas sobre una escena completa.

Para una sola imagen: `.\.venv\Scripts\python.exe scripts\predict.py --image demostraciones/imagenes/helmet.jpg --output demostraciones/resultados/helmet.png`. Volver a ejecutar una demostracion actualiza sus salidas del mismo nombre; no modifica el modelo ni las metricas finales.

## Estado

Comprobacion tecnica: `scripts/train.py --smoke-test` ejecuta una prueba breve con muestras de entrenamiento y validacion. Comprueba ajuste del clasificador, guardado, recuperacion del modelo y graficos. Guarda todo en `work/verification/`, no consulta test y no publica un modelo final. Sus metricas no se usan en el informe.

El entrenamiento guarda el mejor modelo por perdida de validacion y un checkpoint de la ultima epoca con el estado del optimizador. Un archivo de bloqueo evita ejecuciones simultaneas; si el proceso termina abruptamente, verificar que ya no esta activo antes de retirar `.training.lock`. No se reanuda automaticamente un proceso interrumpido.

Entrenamiento definitivo completado el 19 de octubre de 2026: 9 epocas ejecutadas, modelo de la epoca 5 seleccionado por menor perdida de validacion. Exactitud en prueba: 97.69%; F1 macro: 97.25%. Resultados en `outputs/resultados_finales/` (ejecucion original `run_20261019_102743_25844`), modelo en `models/densenet121_ppe.pt` y resumen en `report/RESULTADOS_VERIFICADOS.md`. No es necesario volver a entrenar para ejecutar la demostracion.
