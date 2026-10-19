import cv2
import sys
from ultralytics import YOLO

def main():
    # Ruta del modelo recién entrenado (YOLO11n)
    # model_path = r"C:\Users\nilso\Documents\ciclo x\VISION-GRAFICOS\DATASET-VISION-PEXEL\runs\detect\train-3\weights\best.pt"
    model_path = r"C:\Users\nilso\Documents\ciclo x\VISION-GRAFICOS\PROYECTO_PPE\modelo_ppe_final.pt"
    
    print("Cargando el súper modelo de Seguridad YOLO11n...")
    try:
        model = YOLO(model_path)
        print("✅ Modelo cargado con éxito.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        sys.exit(1)

    # Configuración de DroidCam (0, 1 o 2)
    camera_index = 0 
    
    print(f"Iniciando DroidCam/Webcam (Índice {camera_index})...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Intentando con el índice 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("❌ No se detectó ninguna cámara.")
            sys.exit(1)

    # Resolución HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "="*50)
    print("🛡️ SISTEMA DE SEGURIDAD INDUSTRIAL ACTIVO 🛡️")
    print("Apunta tu teléfono a ti mismo o a otra persona.")
    print("Ponte/Quítate una gorra/casco o casaca para ver la magia.")
    print("Presiona la tecla 'q' para SALIR.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Pasar frame al modelo
        results = model(frame, conf=0.6, verbose=False) # 60% de seguridad mínima
        
        # Colores personalizados (BGR)
        # 0: Head (Rojo), 1: Helmet (Verde), 2: Vest (Amarillo), 3: Worker (Azul)
        
        # Copiamos el frame para dibujar encima
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Coordenadas
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                label = ""
                color = (255, 255, 255) # Blanco por defecto
                
                if cls == 0: # Head (Sin casco)
                    label = f"PELIGRO: Sin Casco ({conf*100:.0f}%)"
                    color = (0, 0, 255) # Rojo
                elif cls == 1: # Helmet
                    label = f"Casco Seguro ({conf*100:.0f}%)"
                    color = (0, 255, 0) # Verde
                elif cls == 2: # Vest
                    label = f"Chaleco ({conf*100:.0f}%)"
                    color = (0, 200, 255) # Amarillo/Naranja
                elif cls == 3: # Worker
                    label = f"Trabajador ({conf*100:.0f}%)"
                    color = (255, 0, 0) # Azul
                
                # Dibujar rectángulo grueso
                grosor = 3 if cls == 0 else 2 # Más grueso si es peligro
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, grosor)
                
                # Fondo del texto
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                
                # Texto
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Mostrar en pantalla
        cv2.imshow("SafeGuard IA - Prueba PC", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
