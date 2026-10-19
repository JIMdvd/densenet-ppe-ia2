"""Clasifica un recorte usando el modelo guardado, sin volver a entrenar."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import densenet121

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path, help="Imagen recortada del objeto.")
    parser.add_argument("--model", type=Path, default=ROOT / "models/densenet121_ppe.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Grafico opcional de la prediccion.")
    args = parser.parse_args()
    if not args.image.is_file() or not args.model.is_file():
        parser.error("No se encuentra la imagen o el modelo. Revisar las rutas.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA no disponible; usar --device cpu.")
    torch.set_num_threads(4)
    saved = torch.load(args.model, map_location="cpu", weights_only=True)
    config = saved["config"]
    classes = config["classes"]
    size = config["image_size"]
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(classes))
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    with torch.inference_mode():
        scores = model(transform(image).unsqueeze(0).to(device)).softmax(1)[0].cpu()
    if not torch.isfinite(scores).all():
        raise RuntimeError("El modelo produjo valores no finitos.")
    prediction = classes[int(scores.argmax())]
    result = {
        "image": str(args.image.resolve()),
        "prediction": prediction,
        "softmax_scores": dict(zip(classes, scores.tolist())),
        "model_epoch": saved["epoch"],
        "software_test_model": config.get("smoke_test", False),
        "scope": "Clasificacion de un recorte; no detecta objetos ni certifica seguridad.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title(f"Prediccion: {prediction}")
        axes[1].barh(classes, scores.tolist())
        axes[1].set(xlim=(0, 1), xlabel="Puntuacion softmax (no calibrada)")
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
