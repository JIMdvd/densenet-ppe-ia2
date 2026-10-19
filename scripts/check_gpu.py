"""Comprueba PyTorch y una operacion real sobre CUDA sin iniciar entrenamiento."""
import json
import platform
from pathlib import Path

import torch


def main():
    report = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if report["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        report.update(gpu=props.name, memory_gib=round(props.total_memory / 2**30, 2))
        x = torch.ones((256, 256), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        report["cuda_operation_ok"] = y[0, 0].item() == 256
    target = Path(__file__).resolve().parents[1] / "outputs" / "environment.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report.get("cuda_operation_ok"):
        raise SystemExit("CUDA no esta lista. No iniciar entrenamiento aun.")


if __name__ == "__main__":
    main()
