"""Entrena el clasificador final de DenseNet121 y evalua una sola vez en test."""
import argparse
import csv
import json
import random
import time
import math
import os
import hashlib
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import DenseNet121_Weights, densenet121

ROOT = Path(__file__).resolve().parents[1]


def save_checkpoint(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sample_per_class(dataset, limit):
    counts = {}
    selected = []
    for path, label in dataset.samples:
        if counts.get(label, 0) < limit:
            selected.append((path, label))
            counts[label] = counts.get(label, 0) + 1
    dataset.samples = selected
    dataset.imgs = selected
    dataset.targets = [label for _, label in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=0, help="Epocas sin mejora suficiente antes de parar; 0 desactiva.")
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true", help="Prueba corta aislada; no usa el conjunto test ni publica un modelo final.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("Revisar epochs, batch-size e image-size.")
    if args.patience < 0 or args.min_epochs < 1 or not math.isfinite(args.min_delta) or args.min_delta < 0:
        raise SystemExit("Revisar patience, min-epochs y min-delta.")
    if args.smoke_test:
        args.epochs = 1
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no disponible. Ejecutar primero scripts/check_gpu.py.")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.hub.set_dir(str(ROOT / ".cache" / "torch"))
    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    data = {split: datasets.ImageFolder(ROOT / "data" / "classification" / ("valid" if args.smoke_test and split == "test" else split), transform=transform) for split in ("train", "valid", "test")}
    if args.smoke_test:
        for split, dataset in data.items():
            sample_per_class(dataset, 4 if split == "train" else 2)
    if not (data["train"].class_to_idx == data["valid"].class_to_idx == data["test"].class_to_idx):
        raise ValueError("Las clases no coinciden entre conjuntos.")
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"), num_workers=0, pin_memory=True) for split, ds in data.items()}
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, len(data["train"].classes))
    model.to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    run = ROOT / ("work/verification" if args.smoke_test else "outputs") / (time.strftime("run_%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    run.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), seed=seed, model="DenseNet121", pretrained_weights="IMAGENET1K_V1", training_mode="frozen_backbone_train_classifier", classes=data["train"].classes, class_to_idx=data["train"].class_to_idx, gpu=torch.cuda.get_device_name(0), pytorch=torch.__version__, torchvision_input_normalization="ImageNet", train_samples=len(data["train"]), valid_samples=len(data["valid"]), test_samples=len(data["test"]), augmentation=False)
    config["pytorch"] = str(torch.__version__)
    config["evaluation_split"] = "valid_subset_for_software_check" if args.smoke_test else "test"
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["total_parameters"] = sum(p.numel() for p in model.parameters())
    manifest = ROOT / "outputs/dataset_manifest.csv"
    config["dataset_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for filename in ("dataset_manifest.csv", "dataset_summary.json"):
        shutil.copy2(ROOT / "outputs" / filename, run / filename)
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"GPU: {config['gpu']} | clases: {config['classes']} | resultados: {run}", flush=True)
    history = []
    best_loss = float("inf")
    stopping_reference = float("inf")
    epochs_without_improvement = 0
    best_path = run / "best_densenet121.pt"
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        metrics = {"epoch": epoch}
        for split in ("train", "valid"):
            # BatchNorm del backbone permanece en evaluacion al estar congelado.
            model.eval()
            model.classifier.train(split == "train")
            loss_sum, correct, total = 0.0, 0, 0
            with torch.set_grad_enabled(split == "train"):
                for batch_index, (images, targets) in enumerate(loaders[split], 1):
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Perdida no finita en {split}, epoca {epoch}, lote {batch_index}.")
                    if split == "train":
                        loss.backward()
                        optimizer.step()
                    loss_sum += loss.item() * targets.size(0)
                    correct += (logits.argmax(1) == targets).sum().item()
                    total += targets.size(0)
                    if split == "train" and batch_index % 100 == 0:
                        print(f"Epoca {epoch}: lote {batch_index}/{len(loaders[split])}", flush=True)
            metrics[f"{split}_loss"] = loss_sum / total
            metrics[f"{split}_accuracy"] = correct / total
        torch.cuda.synchronize()
        metrics["seconds"] = time.perf_counter() - started
        history.append(metrics)
        with (run / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerows(history)
        if metrics["valid_loss"] < best_loss:
            best_loss = metrics["valid_loss"]
            save_checkpoint({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, best_path)
        if metrics["valid_loss"] < stopping_reference - args.min_delta:
            stopping_reference = metrics["valid_loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        save_checkpoint({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config, "epoch": epoch, "history": history}, run / "last_checkpoint.pt")
        print(f"Epoca {epoch}/{args.epochs}: loss={metrics['train_loss']:.4f}, accuracy={metrics['train_accuracy']:.3f}, val_loss={metrics['valid_loss']:.4f}, val_accuracy={metrics['valid_accuracy']:.3f}, segundos={metrics['seconds']:.1f}", flush=True)
        if epoch == 1:
            print(f"Estimacion restante basada en primera epoca: {(args.epochs-1)*metrics['seconds']/60:.1f} minutos, mas evaluacion.", flush=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, measure, title in zip(axes, ("loss", "accuracy"), ("Perdida", "Exactitud")):
            for split, label in (("train", "Entrenamiento"), ("valid", "Validacion")):
                ax.plot([h["epoch"] for h in history], [h[f"{split}_{measure}"] for h in history], marker="o", label=label)
            ax.set(xlabel="Epoca", ylabel=title)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run / "learning_curves.png", dpi=160)
        plt.close(fig)
        if args.patience and epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(f"Parada temprana: {args.patience} epocas sin reducir val_loss al menos {args.min_delta}.", flush=True)
            break

    saved = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    actual, predicted, confidence = [], [], []
    with torch.no_grad():
        for images, targets in loaders["test"]:
            logits = model(images.to(device, non_blocking=True))
            actual.extend(targets.tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
            confidence.extend(logits.softmax(1).max(1).values.cpu().tolist())
    labels = list(range(len(data["train"].classes)))
    report = classification_report(actual, predicted, labels=labels, target_names=data["train"].classes, output_dict=True, zero_division=0)
    majority_class = int(np.bincount(data["train"].targets).argmax())
    report["majority_baseline_accuracy"] = sum(y == majority_class for y in actual) / len(actual)
    report["majority_baseline_class"] = data["train"].classes[majority_class]
    report["balanced_accuracy"] = balanced_accuracy_score(actual, predicted)
    report["smoke_test"] = args.smoke_test
    report["selected_epoch"] = saved["epoch"]
    report["completed_epochs"] = len(history)
    (run / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (run / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["crop", "actual", "predicted", "softmax_score"])
        for (path, _), y, yp, score in zip(data["test"].samples, actual, predicted, confidence):
            writer.writerow([str(Path(path).relative_to(ROOT)), data["test"].classes[y], data["test"].classes[yp], score])
    cm = confusion_matrix(actual, predicted, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=data["train"].classes).plot(ax=ax, colorbar=False)
    ax.set(xlabel="Prediccion", ylabel="Etiqueta real", title="DenseNet121 en conjunto de prueba")
    fig.tight_layout()
    fig.savefig(run / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    errors = [i for i, (y, yp) in enumerate(zip(actual, predicted)) if y != yp]
    correct = [i for i, (y, yp) in enumerate(zip(actual, predicted)) if y == yp]
    selected = (errors[:6] + correct[:6])[:12]
    fig, axes = plt.subplots(max(1, math.ceil(len(selected) / 4)), 4, figsize=(12, max(3, 3 * math.ceil(len(selected) / 4))), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, selected):
        with Image.open(data["test"].samples[index][0]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"Real: {data['train'].classes[actual[index]]}\nPred.: {data['train'].classes[predicted[index]]}", fontsize=10)
    fig.suptitle("Ejemplos de clasificacion: errores y aciertos disponibles")
    fig.tight_layout()
    fig.savefig(run / "prediction_examples.png", dpi=160)
    plt.close(fig)
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    if not args.smoke_test:
        shutil.copy2(best_path, model_dir / "densenet121_ppe.pt")
    if args.smoke_test:
        print(f"COMPROBACION COMPLETADA. Se reutilizo una muestra de validacion; estas metricas no son resultados del experimento. Archivos: {run}", flush=True)
    else:
        print(f"FINALIZADO. Accuracy test: {report['accuracy']:.4f}. Macro F1: {report['macro avg']['f1-score']:.4f}. Resultados: {run}", flush=True)


if __name__ == "__main__":
    lock = ROOT / ".training.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise SystemExit("Ya existe .training.lock. Comprobar si hay otro entrenamiento activo antes de iniciar.")
    try:
        main()
    finally:
        lock.unlink(missing_ok=True)
