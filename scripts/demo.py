"""Ejecuta cuatro ejemplos con el modelo guardado. No entrena."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for label in ("head", "helmet", "vest", "worker"):
        print(f"\nEjemplo etiquetado como: {label}", flush=True)
        subprocess.run([
            sys.executable, "-X", "utf8", str(ROOT / "scripts/predict.py"),
            "--image", str(ROOT / f"demostraciones/imagenes/{label}.jpg"),
            "--output", str(ROOT / f"demostraciones/resultados/{label}.png"),
            "--json-output", str(ROOT / f"demostraciones/resultados/{label}.json"),
        ], cwd=ROOT, check=True)
    print("\nDemostracion terminada. Abrir demostraciones/resultados.")


if __name__ == "__main__":
    main()
