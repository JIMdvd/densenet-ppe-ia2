# Clasificacion de regiones de seguridad en obras con DenseNet121

Proyecto del curso Inteligencia Artificial II. La tarea consiste en clasificar recortes de cuatro categorias: cabeza (Head), casco (Helmet), chaleco (Vest) y trabajador (Worker).

## Organizacion

- `data/raw/`: copia de las 2000 imagenes originales y sus cajas etiquetadas.
- `data/classification/`: recortes organizados por conjunto y categoria para DenseNet.
- `scripts/`: preparacion de datos y comprobacion de GPU.
- `outputs/`: conteos, manifiesto de recortes y ejemplos visuales.
- `references/yolo/`: configuracion y resultados del entrenamiento YOLO anterior, como antecedente.
- `models/`: destino previsto para el modelo DenseNet entrenado.
- `report/` y `slides/`: espacios para informe y exposicion.
- `.venv/`: entorno de Python de este proyecto.

## Datos y alcance

Origen: PPE Compliance Detection, version 5, Roboflow, licencia CC BY 4.0.
https://universe.roboflow.com/yihong-chen/ppe-compliance-detection-iln63/dataset/5

Separacion original: 1400 imagenes train, 400 valid y 200 test. Se conserva esta separacion antes de generar recortes. Las imagenes identicas entre conjuntos, si existen, se excluyen de los conjuntos con menor prioridad para evitar contaminacion (prioridad test, valid, train). El manifiesto conserva la trazabilidad y el resumen registra exclusiones.

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

## Estado

Comprobacion tecnica: `scripts/train.py --smoke-test` ejecuta una prueba breve con muestras de entrenamiento y validacion. Comprueba ajuste del clasificador, guardado, recuperacion del modelo y graficos. Guarda todo en `work/verification/`, no consulta test y no publica un modelo final. Sus metricas no se usan en el informe.

El entrenamiento guarda el mejor modelo por perdida de validacion y un checkpoint de la ultima epoca con el estado del optimizador. Un archivo de bloqueo evita ejecuciones simultaneas; si el proceso termina abruptamente, verificar que ya no esta activo antes de retirar `.training.lock`. No se reanuda automaticamente un proceso interrumpido.

La organizacion y preparacion no equivalen a un entrenamiento. Los resultados experimentales de DenseNet se registraran al ejecutar su entrenamiento y evaluacion.
