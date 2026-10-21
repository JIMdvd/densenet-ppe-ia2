# DenseNet121: implementos de seguridad en obras

Trabajo del curso Inteligencia Artificial II de la UNAMBA.

Estudiante: Jimmy Vargas Huarcaya  
Docente: Ing. Manuel Ibarra Cabrera

En este trabajo utilicé DenseNet121 para clasificar imágenes recortadas de cabeza, casco, chaleco y trabajador. Aproveché imágenes y etiquetas de un proyecto anterior con YOLO para preparar los recortes y entrené el clasificador con mi GTX 1650 Ti.

El entrenamiento terminó en 9 épocas y el mejor resultado de validación fue en la época 5. En la prueba final, el modelo acertó 1.056 de 1.081 imágenes, con una exactitud de 97,69 %. Trabaja con objetos recortados y puede cometer errores.

## Contenido

- `scripts/`: código del proyecto.
- `demostraciones/`: imágenes de ejemplo y sus resultados.
- `outputs/`: datos del entrenamiento, resultados y gráficas.

## Probar el modelo

Con el entorno de Python preparado, descargar el [modelo entrenado](https://github.com/JIMdvd/densenet-ppe-ia2/releases/tag/v1.0.0) y guardar el archivo `densenet121_ppe.pt` en la carpeta `models/`. Desde la terminal del proyecto ejecutar:

```powershell
.\.venv\Scripts\python.exe scripts/demo.py
```

Los resultados se guardan en `demostraciones/resultados/`.

Para probar la cámara en el centro de la imagen, instala las dependencias con `pip install -r requirements.txt` y ejecuta:

```powershell
.\.venv\Scripts\python.exe scripts/demo_camara.py
```

Esta prueba clasifica la región central de la cámara. Los recuadros de detección están en la demo para PC de la carpeta `DenseNet-PPE-PC-Demo`.

Dataset: [PPE Compliance Detection v5](https://universe.roboflow.com/yihong-chen/ppe-compliance-detection-iln63/dataset/5), de Yihong Chen, licencia CC BY 4.0.
