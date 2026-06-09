"""
Sprint 2 - Inferencia y conteo en tiempo real para Raspberry Pi 3B+.

Este script ejecuta un modelo YOLOv8 Nano Segmentacion exportado a ONNX
sin importar ultralytics. Esta pensado para correr por SSH en entorno headless.

Clases:
    0: hoja
    1: palo
"""

from __future__ import annotations

import argparse
import csv
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = {0: "hoja", 1: "palo"}


@dataclass(frozen=True)
class InferenceConfig:
    """Configuracion principal de inferencia."""

    model_path: Path
    camera_index: int
    input_size: int
    confidence_threshold: float
    iou_threshold: float
    mask_threshold: float
    max_detections: int
    camera_width: int
    camera_height: int
    camera_fps: int
    log_path: Path


@dataclass
class LetterboxInfo:
    """Datos necesarios para volver de coordenadas 640x640 al frame original."""

    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int


@dataclass
class Detection:
    """Resultado decodificado de una instancia segmentada."""

    class_id: int
    confidence: float
    box_xyxy: np.ndarray
    contour: Optional[np.ndarray]


class ThreadedCamera:
    """
    Captura frames en un hilo secundario.

    La inferencia consume siempre el ultimo frame disponible. Esto evita que la
    camara quede bloqueada mientras ONNXRuntime procesa el frame anterior.
    """

    def __init__(self, camera_index: int, width: int, height: int, fps: int) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.capture: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "ThreadedCamera":
        """Inicializa la camara y comienza la captura asincronica."""
        try:
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(self.camera_index)

            if not self.capture.isOpened():
                raise RuntimeError(f"No se pudo abrir la camara index={self.camera_index}")

            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print(f"[OK] Camara inicializada en index={self.camera_index}")
            return self
        except Exception as exc:
            self.stop()
            raise RuntimeError(f"Fallo al inicializar la camara: {exc}") from exc

    def _update_loop(self) -> None:
        """Lee frames continuamente y conserva solo el mas reciente."""
        while self.running and self.capture is not None:
            grabbed, frame = self.capture.read()
            if not grabbed:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame

    def read(self) -> Optional[np.ndarray]:
        """Devuelve una copia del ultimo frame disponible."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self) -> None:
        """Detiene el hilo y libera la camara."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()


class YOLOv8SegONNX:
    """Inferencia y decodificacion manual de YOLOv8 Segmentacion exportado a ONNX."""

    def __init__(
        self,
        model_path: Path,
        input_size: int,
        confidence_threshold: float,
        iou_threshold: float,
        mask_threshold: float,
        max_detections: int,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.mask_threshold = mask_threshold
        self.max_detections = max_detections

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"No existe el modelo ONNX: {self.model_path}")

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 2
            session_options.inter_op_num_threads = 1

            # En Raspberry Pi se usa CPUExecutionProvider. No se carga CUDA ni Ultralytics.
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"[OK] Modelo ONNX cargado: {self.model_path}")
            print(f"[OK] Backend ONNXRuntime: {self.session.get_providers()}")
        except Exception as exc:
            raise RuntimeError(f"Fallo al cargar el modelo ONNX: {exc}") from exc

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Ejecuta inferencia sobre un frame BGR y devuelve detecciones segmentadas."""
        input_tensor, info = self._preprocess(frame_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        predictions, prototypes = self._split_outputs(outputs)
        return self._postprocess(predictions, prototypes, info)

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        """Aplica letterbox, BGR->RGB y normalizacion 0..1."""
        original_h, original_w = frame_bgr.shape[:2]
        scale = min(self.input_size / original_w, self.input_size / original_h)
        resized_w = int(round(original_w * scale))
        resized_h = int(round(original_h * scale))

        resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)

        pad_x = (self.input_size - resized_w) // 2
        pad_y = (self.input_size - resized_h) // 2
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        info = LetterboxInfo(
            scale=scale,
            pad_x=float(pad_x),
            pad_y=float(pad_y),
            original_width=original_w,
            original_height=original_h,
        )
        return tensor, info

    @staticmethod
    def _split_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Identifica la salida de predicciones y la salida proto de YOLOv8-seg.

        Formatos comunes:
        - predicciones: [1, 4 + num_clases + 32, 8400] o [1, 8400, 4 + num_clases + 32]
        - prototipos: [1, 32, 160, 160]
        """
        if len(outputs) < 2:
            raise RuntimeError("El modelo ONNX no expuso las dos salidas esperadas: pred y proto")

        first, second = outputs[0], outputs[1]
        if first.ndim == 4:
            proto, pred = first, second
        elif second.ndim == 4:
            pred, proto = first, second
        else:
            raise RuntimeError("No se encontro salida de prototipos de mascara con dimension 4")

        pred = np.squeeze(pred, axis=0)
        proto = np.squeeze(proto, axis=0)

        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        return pred.astype(np.float32, copy=False), proto.astype(np.float32, copy=False)

    def _postprocess(
        self,
        predictions: np.ndarray,
        prototypes: np.ndarray,
        info: LetterboxInfo,
    ) -> list[Detection]:
        """Decodifica cajas, clases, NMS, mascaras y contornos."""
        num_classes = len(CLASS_NAMES)
        mask_coeff_start = 4 + num_classes

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:mask_coeff_start]
        mask_coefficients = predictions[:, mask_coeff_start:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = confidences >= self.confidence_threshold

        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]
        mask_coefficients = mask_coefficients[keep]

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - info.pad_x) / info.scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - info.pad_y) / info.scale
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, info.original_width - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, info.original_height - 1)

        nms_indices = self._class_aware_nms(boxes_xyxy, confidences, class_ids)
        if not nms_indices:
            return []

        nms_indices = nms_indices[: self.max_detections]
        boxes_xyxy = boxes_xyxy[nms_indices]
        class_ids = class_ids[nms_indices]
        confidences = confidences[nms_indices]
        mask_coefficients = mask_coefficients[nms_indices]

        contours = self._build_contours(mask_coefficients, prototypes, boxes_xyxy, info)

        return [
            Detection(
                class_id=int(class_ids[i]),
                confidence=float(confidences[i]),
                box_xyxy=boxes_xyxy[i],
                contour=contours[i],
            )
            for i in range(len(nms_indices))
        ]

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        """Convierte cajas centro-x, centro-y, ancho, alto a x1, y1, x2, y2."""
        boxes = boxes_xywh.copy()
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        return boxes

    def _class_aware_nms(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
    ) -> list[int]:
        """Aplica NMS por clase usando OpenCV para reducir computo en ARM."""
        selected: list[int] = []

        for class_id in np.unique(class_ids):
            indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes_xyxy[indices]
            class_scores = confidences[indices]

            # cv2.dnn.NMSBoxes espera [x, y, w, h].
            boxes_xywh = []
            for box in class_boxes:
                x1, y1, x2, y2 = box
                boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

            nms = cv2.dnn.NMSBoxes(
                boxes_xywh,
                class_scores.astype(float).tolist(),
                self.confidence_threshold,
                self.iou_threshold,
            )

            if len(nms) == 0:
                continue

            for nms_id in np.array(nms).flatten():
                selected.append(int(indices[nms_id]))

        selected.sort(key=lambda idx: float(confidences[idx]), reverse=True)
        return selected

    def _build_contours(
        self,
        mask_coefficients: np.ndarray,
        prototypes: np.ndarray,
        boxes_xyxy: np.ndarray,
        info: LetterboxInfo,
    ) -> list[Optional[np.ndarray]]:
        """
        Combina coeficientes y prototipos para obtener mascaras binarias.

        YOLOv8-seg produce una base de prototipos [32, Hm, Wm]. Cada deteccion
        trae coeficientes que ponderan esos prototipos. La multiplicacion
        coeficientes @ prototipos reconstruye la mascara de cada instancia.
        """
        proto_channels, proto_h, proto_w = prototypes.shape
        proto_flat = prototypes.reshape(proto_channels, -1)
        masks = self._sigmoid(mask_coefficients @ proto_flat)
        masks = masks.reshape(-1, proto_h, proto_w)

        contours: list[Optional[np.ndarray]] = []
        for mask, box in zip(masks, boxes_xyxy):
            full_mask = cv2.resize(
                mask,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_LINEAR,
            )

            x_start = int(round(info.pad_x))
            y_start = int(round(info.pad_y))
            valid_w = int(round(info.original_width * info.scale))
            valid_h = int(round(info.original_height * info.scale))
            unpadded = full_mask[y_start : y_start + valid_h, x_start : x_start + valid_w]

            if unpadded.size == 0:
                contours.append(None)
                continue

            original_mask = cv2.resize(
                unpadded,
                (info.original_width, info.original_height),
                interpolation=cv2.INTER_LINEAR,
            )

            binary_mask = (original_mask >= self.mask_threshold).astype(np.uint8) * 255
            binary_mask = self._crop_mask_to_box(binary_mask, box)
            found_contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not found_contours:
                contours.append(None)
                continue

            largest = max(found_contours, key=cv2.contourArea)
            if cv2.contourArea(largest) <= 0:
                contours.append(None)
            else:
                contours.append(largest)

        return contours

    @staticmethod
    def _crop_mask_to_box(mask: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        """Elimina pixeles de mascara fuera de la caja detectada."""
        x1, y1, x2, y2 = box_xyxy.astype(int)
        cropped = np.zeros_like(mask)
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return cropped

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        """Sigmoide numericamente estable para mapas de mascara."""
        values = np.clip(values, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-values))


class CSVLogger:
    """Registro persistente de conteos por frame."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.log_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        if self.log_path.stat().st_size == 0:
            self.writer.writerow(["timestamp", "hojas", "palos", "fps"])
            self.file.flush()

    def write(self, hojas: int, palos: int, fps: float) -> None:
        """Agrega una fila al CSV."""
        timestamp = datetime.now().isoformat(timespec="seconds")
        self.writer.writerow([timestamp, hojas, palos, f"{fps:.2f}"])
        self.file.flush()

    def close(self) -> None:
        """Cierra el archivo de log."""
        self.file.close()


def count_classes(detections: list[Detection]) -> tuple[int, int]:
    """Cuenta hojas y palos a partir de detecciones validas."""
    hojas = sum(1 for detection in detections if detection.class_id == 0)
    palos = sum(1 for detection in detections if detection.class_id == 1)
    return hojas, palos


def parse_args() -> InferenceConfig:
    """Parsea argumentos CLI para uso flexible por SSH."""
    parser = argparse.ArgumentParser(
        description="Inferencia YOLOv8-seg ONNX para conteo de hoja/palo en Raspberry Pi."
    )
    parser.add_argument("--model", type=Path, default=Path("best.onnx"), help="Ruta a best.onnx")
    parser.add_argument("--camera", type=int, default=0, help="Indice de camara OpenCV")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de entrada del modelo")
    parser.add_argument("--conf", type=float, default=0.35, help="Umbral de confianza")
    parser.add_argument("--iou", type=float, default=0.45, help="Umbral IoU para NMS")
    parser.add_argument("--mask-thres", type=float, default=0.50, help="Umbral binario de mascara")
    parser.add_argument("--max-det", type=int, default=100, help="Maximo de detecciones por frame")
    parser.add_argument("--cam-width", type=int, default=640, help="Ancho solicitado a la camara")
    parser.add_argument("--cam-height", type=int, default=480, help="Alto solicitado a la camara")
    parser.add_argument("--cam-fps", type=int, default=15, help="FPS solicitado a la camara")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("Docs") / "inferencia_sprint2_log.csv",
        help="Archivo CSV de salida",
    )
    args = parser.parse_args()

    return InferenceConfig(
        model_path=args.model,
        camera_index=args.camera,
        input_size=args.imgsz,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        mask_threshold=args.mask_thres,
        max_detections=args.max_det,
        camera_width=args.cam_width,
        camera_height=args.cam_height,
        camera_fps=args.cam_fps,
        log_path=args.log,
    )


def main() -> None:
    """Loop principal headless: captura, inferencia, conteo, FPS y CSV."""
    config = parse_args()
    stop_event = threading.Event()

    def handle_signal(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    camera: Optional[ThreadedCamera] = None
    logger: Optional[CSVLogger] = None

    try:
        model = YOLOv8SegONNX(
            model_path=config.model_path,
            input_size=config.input_size,
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            mask_threshold=config.mask_threshold,
            max_detections=config.max_detections,
        )
        camera = ThreadedCamera(
            camera_index=config.camera_index,
            width=config.camera_width,
            height=config.camera_height,
            fps=config.camera_fps,
        ).start()
        logger = CSVLogger(config.log_path)

        print("[INFO] Inferencia iniciada. Presione Ctrl+C para finalizar.")
        frame_number = 0

        while not stop_event.is_set():
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            start_time = time.perf_counter()
            detections = model.infer(frame)
            elapsed = time.perf_counter() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            hojas, palos = count_classes(detections)
            frame_number += 1

            print(
                f"[FRAME {frame_number}] Hojas detectadas: {hojas} | "
                f"Palos detectados: {palos} | FPS: {fps:.1f}",
                flush=True,
            )
            logger.write(hojas, palos, fps)

    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
    finally:
        if camera is not None:
            camera.stop()
        if logger is not None:
            logger.close()
        print("[INFO] Recursos liberados. Fin de ejecucion.", flush=True)


if __name__ == "__main__":
    main()
