# Resultados verificados de DenseNet121

Ejecucion definitiva: `outputs/run_20261019_102743_25844`.
Modelo: `models/densenet121_ppe.pt`.

## Experimento

Clasificacion de regiones anotadas en cuatro clases: head, helmet, vest y worker. Se utilizo DenseNet121 preentrenada en ImageNet, con base congelada y clasificador nuevo de cuatro salidas. Se ajustaron 4 100 parametros de los 6 957 956 parametros totales. No se entreno la red completa desde cero.

Se reutilizaron las cajas anotadas del dataset PPE Compliance Detection v5 para generar recortes, sin ejecutar YOLO. Se excluyeron 30 imagenes duplicadas o casi identicas entre conjuntos y un recorte demasiado pequeno. Los originales se conservaron.

- Entrenamiento: 5 596 recortes.
- Validacion: 1 816 recortes.
- Prueba: 1 081 recortes.
- Entrada: RGB, 128 x 128, normalizacion ImageNet; sin aumentos de datos.
- Optimizador: Adam, tasa de aprendizaje 0.0005, lotes de 8.
- Perdida: entropia cruzada, no MSE.
- Equipo: NVIDIA GTX 1650 Ti; PyTorch 2.7.1+cu118.
- Semilla: 42. Maximo: 20 epocas. Parada: 4 epocas sin mejora de al menos 0.001 en perdida de validacion.

## Seleccion y prueba

Se ejecutaron 9 epocas y se selecciono la epoca 5 por menor perdida de validacion (0.056902). Su exactitud de validacion fue 97.96%. El tiempo acumulado de entrenamiento y validacion fue 6 minutos 23 segundos, sin incluir carga inicial, guardado, graficos ni evaluacion final.

El modelo seleccionado acerto 1 056 de 1 081 recortes de prueba, con 25 errores:

| Medida | Resultado |
|---|---:|
| Exactitud | 97.69% |
| F1 macro | 97.25% |
| Exactitud balanceada | 97.02% |
| Baseline: siempre predecir worker | 36.73% |

| Clase | Precision | Recall | F1 | Recortes |
|---|---:|---:|---:|---:|
| head | 96.59% | 94.44% | 95.51% | 90 |
| helmet | 98.69% | 98.37% | 98.53% | 307 |
| vest | 97.19% | 96.52% | 96.85% | 287 |
| worker | 97.51% | 98.74% | 98.12% | 397 |

La confusion mas frecuente fue vest con worker: 10 chalecos clasificados como trabajador y 5 trabajadores como chaleco. Se comprobaron la carga del modelo exportado y su uso con `scripts/predict.py`.

## Material disponible

En la carpeta de la ejecucion: `learning_curves.png`, `confusion_matrix.png`, `prediction_examples.png`, `demostracion.png`, `history.csv`, `test_metrics.json`, `test_predictions.csv`, `config.json` y copias del manifiesto y resumen de datos.

## Alcance y limitaciones

El resultado corresponde a recortes del dataset, no a reconocimiento perfecto en fotos nuevas. El modelo no localiza objetos ni certifica que una persona use correctamente todos sus equipos de seguridad. Las puntuaciones softmax no estan calibradas como probabilidades de acierto. La clase head tiene menos ejemplos y menor recall. Pueden existir escenas relacionadas no detectadas por el filtro de duplicados; la evaluacion es por recorte, no por obra o escena independiente.

El ensayo previo de cinco epocas se conserva como preliminar y no debe mezclarse con estos resultados. Las metricas de YOLO del proyecto anterior corresponden a deteccion y no son directamente comparables con esta exactitud de clasificacion. Fuentes y procedencia en `references/sources.md`.
