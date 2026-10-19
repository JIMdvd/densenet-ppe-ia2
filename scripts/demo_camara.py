"""Prueba DenseNet121 en vivo con la cámara integrada o USB."""
import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import densenet121

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"head": "Cabeza", "helmet": "Casco", "vest": "Chaleco", "worker": "Trabajador"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara (predeterminado: 0).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model", type=Path, default=ROOT / "models/densenet121_ppe.pt")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"No se encuentra el modelo: {args.model}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA no está disponible. Prueba con --device cpu.")

    torch.set_num_threads(4)
    saved = torch.load(args.model, map_location="cpu", weights_only=True)
    classes = saved["config"]["classes"]
    size = saved["config"]["image_size"]
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(classes))
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        parser.error("No pude abrir la cámara. Revisa el permiso de cámara de Windows o prueba --camera 1.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    label = "Coloca un objeto dentro del recuadro"
    confidence = 0.0
    last_prediction = 0.0
    last_result = None
    print("Cámara abierta. Centra un objeto en el recuadro amarillo. Q: salir | S: guardar captura")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("No se pudo leer la cámara.")
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            side = min(height, width, 640)
            left, top = (width - side) // 2, (height - side) // 2
            crop = frame[top:top + side, left:left + side]
            now = time.monotonic()
            if now - last_prediction >= 0.45:
                image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                with torch.inference_mode():
                    scores = model(transform(image).unsqueeze(0).to(device)).softmax(1)[0].cpu()
                idx = int(scores.argmax())
                label = NAMES.get(classes[idx], classes[idx])
                confidence = float(scores[idx])
                last_result = {"prediction": classes[idx], "label": label,
                               "softmax_scores": dict(zip(classes, scores.tolist())),
                               "model_epoch": saved["epoch"], "device": device,
                               "scope": "Clasificación del recorte central; no detecta objetos en toda la escena."}
                last_prediction = now

            cv2.rectangle(frame, (left, top), (left + side, top + side), (0, 220, 255), 2)
            cv2.rectangle(frame, (18, 18), (760, 96), (20, 20, 20), -1)
            cv2.putText(frame, f"{label}  |  puntuacion: {confidence:.1%}", (34, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Q: salir   S: guardar resultado", (34, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow("Prueba DenseNet121 | centra un objeto en el recuadro", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and last_result:
                out = ROOT / "demostraciones/resultados/camara"
                out.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(out / f"captura_{stamp}.jpg"), crop)
                (out / f"captura_{stamp}.json").write_text(
                    json.dumps(last_result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"Guardado en {out.relative_to(ROOT)} | {label} ({confidence:.1%})")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
