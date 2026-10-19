"""Copia el dataset original y convierte sus cajas YOLO en recortes de clasificacion."""
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {0: "head", 1: "helmet", 2: "vest", 3: "worker"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def duplicate_exclusions(raw):
    records = []
    for split in ("test", "valid", "train"):
        for path in sorted((raw / split / "images").iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                small = np.asarray(rgb.convert("L").resize((9, 8)), dtype=np.int16)
                bits = (small[:, 1:] > small[:, :-1]).ravel()
                dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
                thumbnail = np.asarray(rgb.resize((64, 64)), dtype=np.int16)
            records.append((split, path, hashlib.sha256(path.read_bytes()).hexdigest(), dhash, thumbnail))
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, first in enumerate(records):
        for j in range(i):
            second = records[j]
            exact = first[2] == second[2]
            similar = (first[3] ^ second[3]).bit_count() <= 6
            if exact or (similar and np.abs(first[4] - second[4]).mean() <= 3.0):
                parents[root(i)] = root(j)
    groups = defaultdict(list)
    for index in range(len(records)):
        groups[root(index)].append(index)
    excluded = {}
    for members in groups.values():
        # Los registros ya estan ordenados por prioridad test, valid, train.
        keeper = records[min(members)]
        for index in members:
            record = records[index]
            if record[0] != keeper[0]:
                excluded[(record[0], record[1].name)] = {
                    "split": record[0], "image": record[1].name,
                    "reason": "exact_duplicate_across_splits" if record[2] == keeper[2] else "near_duplicate_across_splits",
                    "kept_split": keeper[0], "kept_image": keeper[1].name,
                }
    return excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "data.yaml").is_file():
        raise SystemExit("La carpeta debe contener data.yaml y train, valid y test.")
    raw = ROOT / "data" / "raw"
    crops = ROOT / "data" / "classification"
    if (ROOT / ".training.lock").exists():
        raise SystemExit("No preparar datos mientras hay un entrenamiento activo.")
    if crops.exists():
        raise SystemExit("classification ya existe. Archivar sus datos antes de regenerar para evitar recortes antiguos.")
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
    duplicates = duplicate_exclusions(raw)
    print(f"Imagenes duplicadas entre conjuntos a excluir: {len(duplicates)}", flush=True)
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
            if (split, image_path.name) in duplicates:
                excluded.append(duplicates[(split, image_path.name)])
                continue
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
    summary["duplicate_filter"] = "SHA256 o dHash Hamming <=6 y diferencia RGB media <=3/255 a 64x64; grupos transitivos; prioridad test > valid > train. No detecta todas las escenas relacionadas."
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
