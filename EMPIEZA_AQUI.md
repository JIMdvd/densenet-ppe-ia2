# Guia del proyecto

El modelo ya esta entrenado. Para presentar o probarlo no hay que entrenar otra vez.

## Que abrir primero

1. `demostraciones/imagenes/`: las entradas de ejemplo.
2. `demostraciones/resultados/`: las respuestas del modelo.
3. `outputs/resultados_finales/graficas/`: las figuras del experimento para el informe.
4. `report/RESULTADOS_VERIFICADOS.md`: explicacion de los resultados reales.

## Mapa

```text
DenseNet-PPE/
  demostraciones/
    imagenes/                 cabeza, casco, chaleco y trabajador
    resultados/               imagenes con prediccion y puntuaciones JSON
  models/
    densenet121_ppe.pt         modelo listo para usar
  outputs/
    resultados_finales/
      graficas/               curvas, matriz de confusion y ejemplos
      metricas/               historial y resultados completos de prueba
      configuracion/          parametros y copia del manifiesto de datos
      checkpoints/            mejor modelo y ultima epoca; no son lo mismo
    datos/                    resumen y manifiesto del dataset actual
    entorno/                  comprobacion de GPU
    pruebas_preliminares/     ensayo anterior, no es el resultado final
    entrenamientos/           se crea si se entrena otra vez
  data/                       originales y recortes; no mover manualmente
  scripts/                    codigo de preparacion, entrenamiento y uso
  report/                     resumen verificado; informe por elaborar
  slides/                     espacio para la presentacion por elaborar
  references/                 fuentes y antecedente YOLO, no DenseNet
  work/                       verificaciones tecnicas y respaldo preliminar
  .venv/                      entorno Python; no mover ni borrar
```

## Demostracion en VS Code

En una terminal abierta en esta carpeta:

```powershell
.\.venv\Scripts\python.exe scripts\demo.py
```

Abrir despues `demostraciones/resultados/head.png`, `helmet.png`, `vest.png` y `worker.png`. Los JSON del mismo nombre conservan las puntuaciones. Las cuatro entradas son ejemplos del conjunto de prueba, no fotos nuevas ni una nueva evaluacion completa. Repetir el comando actualiza esas salidas.

Tambien: Terminal > Ejecutar tarea > Demostracion: cuatro ejemplos (sin entrenar).

Para probar solo el casco:

```powershell
.\.venv\Scripts\python.exe scripts\predict.py --image demostraciones/imagenes/helmet.jpg --output demostraciones/resultados/helmet.png
```

## Que significa cada figura final

- `learning_curves.png`: compara entrenamiento y validacion a lo largo de las epocas.
- `confusion_matrix.png`: muestra aciertos y confusiones de las cuatro clases en prueba.
- `prediction_examples.png`: ejemplos de errores y aciertos; no ocultar los errores al exponer.

Resultado final: 97.69% de exactitud y 97.25% de F1 macro en 1081 recortes. Hubo 25 errores. Se ejecutaron 9 epocas y se selecciono la 5 por validacion.

DenseNet clasifica un recorte; este proyecto no localiza automaticamente todos los objetos de una foto completa. Las puntuaciones softmax no garantizan un acierto. Los antecedentes YOLO y las pruebas preliminares no deben presentarse como resultados finales de DenseNet.
