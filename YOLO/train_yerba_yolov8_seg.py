"""
Sprint 1 - Entrenamiento YOLOv8 Nano Segmentacion para Yerba Mate.

Este script entrena un modelo YOLOv8n-seg con datasets exportados desde CVAT
en formato YOLO y exporta automaticamente el mejor modelo a ONNX para su
posterior despliegue en Raspberry Pi 3B+.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class TrainingConfig:
    """Parametros principales del entrenamiento."""

    base_model: str = "yolov8n-seg.pt"
    data_yaml: Path = Path("data.yaml")
    project_dir: Path = Path("runs/yerba_mate_seg")
    run_name: str = "sprint1_yolov8n_seg"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    workers: int = 8
    patience: int = 30
    seed: int = 42


def resolve_device() -> int | str:
    """
    Verifica si CUDA esta disponible y devuelve el dispositivo de entrenamiento.

    Retorna:
        0 si hay una GPU CUDA disponible.
        "cpu" si no hay GPU CUDA disponible.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"[OK] CUDA disponible. Entrenando con GPU: {gpu_name}")
        print(f"[OK] Version CUDA detectada por PyTorch: {cuda_version}")
        return 0

    print("[AVISO] CUDA no esta disponible. El entrenamiento usara CPU.")
    print("[AVISO] En CPU el entrenamiento puede ser considerablemente mas lento.")
    return "cpu"


def validate_dataset_config(data_yaml: Path) -> None:
    """
    Comprueba que exista el archivo data.yaml generado por CVAT.

    Estructura esperada de data.yaml:

    path: C:/ruta/al/dataset_yerba
    train: images/train
    val: images/val

    names:
      0: hoja
      1: palo

    Tambien es valido usar rutas absolutas en train y val si CVAT exporto el
    dataset de esa manera.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"No se encontro '{data_yaml}'. Coloque el data.yaml exportado por "
            "CVAT en la raiz del proyecto o ajuste TrainingConfig.data_yaml."
        )


def train_model(config: TrainingConfig, device: int | str) -> Path:
    """Entrena YOLOv8n-seg y devuelve la ruta al mejor checkpoint best.pt."""
    print("[INFO] Cargando modelo base YOLOv8 Nano Segmentacion...")
    model = YOLO(config.base_model)

    print("[INFO] Iniciando entrenamiento...")
    results = model.train(
        data=str(config.data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        device=device,
        workers=config.workers,
        project=str(config.project_dir),
        name=config.run_name,
        patience=config.patience,
        seed=config.seed,
        pretrained=True,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(
            f"El entrenamiento finalizo, pero no se encontro el checkpoint: {best_weights}"
        )

    print(f"[OK] Mejor checkpoint encontrado: {best_weights}")
    return best_weights


def export_to_onnx(best_weights: Path, image_size: int) -> Path:
    """
    Exporta best.pt a ONNX.

    ONNX es adecuado para inferencia liviana en CPU/ARM, incluyendo Raspberry Pi.
    El parametro simplify reduce el grafo cuando la dependencia onnxslim esta
    disponible, ayudando al despliegue embebido.
    """
    print("[INFO] Exportando mejor modelo a ONNX...")
    trained_model = YOLO(str(best_weights))
    exported_path = trained_model.export(
        format="onnx",
        imgsz=image_size,
        opset=12,
        simplify=True,
        dynamic=False,
    )

    onnx_path = Path(exported_path)
    print(f"[OK] Modelo exportado a ONNX: {onnx_path}")
    return onnx_path


def main() -> None:
    """Punto de entrada principal del entrenamiento Sprint 1."""
    config = TrainingConfig()

    # Para RTX 4050 se recomienda comenzar con batch=16.
    # Si durante el entrenamiento sobra VRAM, puede probarse batch=32.
    print("[INFO] Configuracion de entrenamiento:")
    print(f"       Modelo base: {config.base_model}")
    print(f"       Dataset YAML: {config.data_yaml}")
    print(f"       Epochs: {config.epochs}")
    print(f"       Batch: {config.batch}")
    print(f"       Imagen: {config.imgsz}x{config.imgsz}")

    validate_dataset_config(config.data_yaml)
    device = resolve_device()
    best_weights = train_model(config, device)
    export_to_onnx(best_weights, config.imgsz)

    print("[OK] Proceso completo: entrenamiento y exportacion ONNX finalizados.")


if __name__ == "__main__":
    main()
