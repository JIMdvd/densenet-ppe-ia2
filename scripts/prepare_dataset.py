"""Copia el dataset original y convierte sus cajas YOLO en recortes de clasificacion."""
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {0: "head", 1: "helmet", 2: "vest", 3: "worker"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "data.yaml").is_file():
        raise SystemExit("La carpeta debe contener data.yaml y train, valid y test.")
    raw = ROOT / "data" / "raw"
    crops = ROOT / "data" / "classification"
    report_dir = ROOT / "outputs"
    report_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        for folder in ("images", "labels"):
            shutil.copytree(source / split / folder, raw / split / folder, dirs_exist_ok=True)
    for name in ("data.yaml", "README.dataset.txt", "README.roboflow.txt"):
        if (source / name).is_file():
            shutil.copy2(source / name, raw / name)

    counts = defaultdict(Counter)
    source_counts = {}
    excluded = []
    seen = {}
    samples = defaultdict(list)
    manifest = []
    # Prioridad de conservacion: prueba, validacion y entrenamiento.
    # Una imagen identica nunca produce recortes en dos conjuntos distintos.
    for split in ("test", "valid", "train"):
        for label in CLASSES.values():
            (crops / split / label).mkdir(parents=True, exist_ok=True)
        images = sorted(p for p in (raw / split / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        source_counts[split] = len(images)
        for image_path in images:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                excluded.append({"split": split, "image": image_path.name, "reason": "exact_duplicate_across_splits", "kept_split": seen[digest]})
                continue
            seen[digest] = split
            label_path = raw / split / "labels" / (image_path.stem + ".txt")
            if not label_path.exists():
                raise ValueError(f"Falta etiqueta: {label_path}")
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                width, height = image.size
                for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError(f"Formato no esperado: {label_path}:{index+1}")
                    category = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    if category not in CLASSES or not all(math.isfinite(v) for v in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                        raise ValueError(f"Caja invalida: {label_path}:{index+1}")
                    x1 = max(0, math.floor((cx - bw / 2) * width))
                    y1 = max(0, math.floor((cy - bh / 2) * height))
                    x2 = min(width, math.ceil((cx + bw / 2) * width))
                    y2 = min(height, math.ceil((cy + bh / 2) * height))
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        excluded.append({"split": split, "image": image_path.name, "box": index, "reason": "crop_smaller_than_4_pixels"})
                        continue
                    label = CLASSES[category]
                    dest = crops / split / label / f"{image_path.stem}__{index:03d}.jpg"
                    image.crop((x1, y1, x2, y2)).save(dest, quality=95)
                    counts[split][label] += 1
                    manifest.append({"split": split, "class": label, "source_image": image_path.name, "source_sha256": digest, "box_index": index, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "crop": str(dest.relative_to(ROOT))})
                    if split == "train" and len(samples[label]) < 5 and x2 - x1 >= 20 and y2 - y1 >= 20:
                        samples[label].append(dest)
        print(f"{split}: {source_counts[split]} imagenes, recortes {dict(counts[split])}", flush=True)

    with (report_dir / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = {"source": str(source), "source_images": source_counts, "crop_counts": {k: dict(v) for k, v in counts.items()}, "total_crops": len(manifest), "classes": CLASSES, "excluded": excluded, "license": "CC BY 4.0", "provenance": "https://universe.roboflow.com/yihong-chen/ppe-compliance-detection-iln63/dataset/5", "task": "Clasificacion de regiones etiquetadas; no certifica cumplimiento de seguridad de una persona o escena."}
    (report_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    canvas = Image.new("RGB", (1000, 760), "white")
    draw = ImageDraw.Draw(canvas)
    for row, label in enumerate(CLASSES.values()):
        draw.text((12, row * 190 + 8), label, fill="black")
        for col, path in enumerate(samples[label]):
            with Image.open(path) as image:
                tile = ImageOps.contain(image.convert("RGB"), (180, 150))
                canvas.paste(tile, (col * 200 + 10, row * 190 + 30))
    canvas.save(report_dir / "dataset_samples.jpg")
    print(f"Total: {len(manifest)} recortes; excluidos: {len(excluded)}. Datos originales preservados.", flush=True)


if __name__ == "__main__":
    main()
