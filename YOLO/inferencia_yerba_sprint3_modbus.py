"""
Sprint 3 - Automatizacion industrial, control de calidad y Modbus TCP.

Este script integra la inferencia ONNX del Sprint 2 con:
- Calculo de porcentaje de palo por frame.
- Filtro de media movil para estabilizar la decision de calidad.
- Escritura sincronica de Holding Registers Modbus TCP hacia PLC Siemens.
- Guardado de imagenes de auditoria cuando una muestra queda RECHAZADA.

Pensado para Raspberry Pi 3B+ en entorno headless por SSH.
"""

from __future__ import annotations

import argparse
import csv
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from inferencia_yerba_sprint2 import ThreadedCamera, YOLOv8SegONNX, count_classes

try:
    # pymodbus 3.x
    from pymodbus.client import ModbusTcpClient
except ImportError:
    try:
        # pymodbus 2.x
        from pymodbus.client.sync import ModbusTcpClient
    except ImportError:
        ModbusTcpClient = None


QUALITY_OK = 0
QUALITY_REJECTED = 1


@dataclass(frozen=True)
class AutomationConfig:
    """Configuracion de inferencia, calidad, auditoria y comunicacion industrial."""

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
    window: int
    max_palo_pct: float
    plc_ip: str
    plc_port: int
    plc_unit_id: int
    plc_reconnect_interval: float
    cooldown: float
    log_path: Path
    rejection_dir: Path


@dataclass(frozen=True)
class QualityResult:
    """Resultado de control de calidad calculado para un frame."""

    raw_palo_pct: float
    filtered_palo_pct: float
    status: int

    @property
    def status_label(self) -> str:
        return "RECHAZADA" if self.status == QUALITY_REJECTED else "OK"


def calculate_palo_percentage(hojas: int, palos: int) -> float:
    """Calcula porcentaje de palo: palos / (hojas + palos) * 100."""
    total = hojas + palos
    if total <= 0:
        return 0.0
    return (palos / total) * 100.0


class MovingAverageQualityFilter:
    """
    Filtro de media movil para evitar decisiones inestables por ruido.

    La ventana se implementa con collections.deque para costo O(1) al agregar
    una muestra y memoria acotada, algo importante en Raspberry Pi 3B+.
    """

    def __init__(self, window: int, max_palo_pct: float) -> None:
        if window <= 0:
            raise ValueError("--window debe ser mayor que 0")
        self.samples: deque[float] = deque(maxlen=window)
        self.max_palo_pct = max_palo_pct

    def update(self, hojas: int, palos: int) -> QualityResult:
        """Agrega el porcentaje actual y devuelve la decision filtrada."""
        raw_pct = calculate_palo_percentage(hojas, palos)
        self.samples.append(raw_pct)
        filtered_pct = sum(self.samples) / len(self.samples)
        status = QUALITY_REJECTED if filtered_pct > self.max_palo_pct else QUALITY_OK
        return QualityResult(
            raw_palo_pct=raw_pct,
            filtered_palo_pct=filtered_pct,
            status=status,
        )


class SiemensModbusClient:
    """
    Cliente Modbus TCP sincronico para PLC Siemens S7-1200/1500.

    Holding Registers escritos desde direccion 0:
    - HR0: hojas frame actual
    - HR1: palos frame actual
    - HR2: porcentaje palo filtrado * 100
    - HR3: estado calidad, 0 OK, 1 RECHAZADO
    """

    def __init__(
        self,
        plc_ip: str,
        plc_port: int,
        unit_id: int,
        reconnect_interval: float,
    ) -> None:
        if ModbusTcpClient is None:
            raise RuntimeError(
                "No se encontro pymodbus. Instalar en Raspberry Pi con: "
                "pip3 install pymodbus"
            )

        self.plc_ip = plc_ip
        self.plc_port = plc_port
        self.unit_id = unit_id
        self.reconnect_interval = reconnect_interval
        self.client: Optional[ModbusTcpClient] = None
        self.last_connect_attempt = 0.0

    def connect(self, force: bool = False) -> bool:
        """Conecta o reconecta respetando un intervalo para no saturar la red."""
        now = time.monotonic()
        if not force and now - self.last_connect_attempt < self.reconnect_interval:
            return self.is_connected()

        self.last_connect_attempt = now
        self.close()

        try:
            self.client = ModbusTcpClient(host=self.plc_ip, port=self.plc_port, timeout=1.0)
        except TypeError:
            # Compatibilidad con pymodbus 2.x.
            self.client = ModbusTcpClient(self.plc_ip, port=self.plc_port, timeout=1.0)

        try:
            connected = bool(self.client.connect())
        except Exception as exc:
            print(f"[PLC] Error conectando a {self.plc_ip}:{self.plc_port}: {exc}", flush=True)
            connected = False

        if connected:
            print(f"[PLC] Conexion Modbus TCP establecida con {self.plc_ip}:{self.plc_port}", flush=True)
        else:
            print(f"[PLC] Sin conexion con PLC {self.plc_ip}:{self.plc_port}", flush=True)
        return connected

    def is_connected(self) -> bool:
        """Evalua conexion activa compatible con pymodbus 2.x y 3.x."""
        if self.client is None:
            return False

        connected_attr = getattr(self.client, "connected", None)
        if isinstance(connected_attr, bool):
            return connected_attr

        is_socket_open = getattr(self.client, "is_socket_open", None)
        if callable(is_socket_open):
            try:
                return bool(is_socket_open())
            except Exception:
                return False

        return False

    def write_quality_registers(self, hojas: int, palos: int, filtered_pct: float, status: int) -> bool:
        """Escribe los cuatro Holding Registers requeridos por el PLC."""
        if not self.is_connected() and not self.connect():
            return False

        registers = [
            self._to_uint16(hojas),
            self._to_uint16(palos),
            self._to_uint16(round(filtered_pct * 100.0)),
            self._to_uint16(status),
        ]

        try:
            assert self.client is not None
            result = self._write_registers_compat(registers)
            has_error = bool(getattr(result, "isError", lambda: False)())
            if has_error:
                print(f"[PLC] Error Modbus al escribir registros: {result}", flush=True)
                self.close()
                return False
            return True
        except Exception as exc:
            print(f"[PLC] Escritura fallida, se intentara reconectar: {exc}", flush=True)
            self.close()
            return False

    def _write_registers_compat(self, registers: list[int]) -> object:
        """Compatibilidad entre firmas de pymodbus 2.x, 3.x y variantes recientes."""
        assert self.client is not None

        try:
            return self.client.write_registers(address=0, values=registers, slave=self.unit_id)
        except TypeError:
            pass

        try:
            return self.client.write_registers(address=0, values=registers, unit=self.unit_id)
        except TypeError:
            pass

        try:
            return self.client.write_registers(address=0, values=registers, device_id=self.unit_id)
        except TypeError:
            pass

        return self.client.write_registers(0, registers)

    def close(self) -> None:
        """Cierra el socket Modbus si esta abierto."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None

    @staticmethod
    def _to_uint16(value: float | int) -> int:
        """Convierte a entero unsigned 16 bits saturado para Holding Register."""
        return max(0, min(65535, int(value)))


class AuditImageSaver:
    """
    Guarda frames de auditoria para muestras rechazadas con cooldown temporal.

    El cooldown evita llenar la SD con multiples imagenes si el mismo material
    defectuoso queda detenido frente a la camara.
    """

    def __init__(self, rejection_dir: Path, cooldown_seconds: float) -> None:
        self.rejection_dir = rejection_dir
        self.cooldown_seconds = cooldown_seconds
        self.last_save_time = 0.0
        self.rejection_dir.mkdir(parents=True, exist_ok=True)

    def save_if_allowed(self, frame_bgr, filtered_palo_pct: float) -> Optional[Path]:
        """Guarda imagen anotada si paso el cooldown; si no, no hace nada."""
        now = time.monotonic()
        if now - self.last_save_time < self.cooldown_seconds:
            return None

        timestamp = datetime.now()
        filename = (
            f"rechazo_{timestamp.strftime('%Y%m%d_%H%M%S')}_"
            f"palo_{filtered_palo_pct:.2f}pct.jpg"
        )
        output_path = self.rejection_dir / filename

        annotated = frame_bgr.copy()
        overlay_text = (
            f"RECHAZADA | {timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Palo filtrado: {filtered_palo_pct:.2f}%"
        )
        cv2.putText(
            annotated,
            overlay_text,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        self.last_save_time = now
        return output_path


class AutomationCSVLogger:
    """CSV industrial con conteos, porcentajes, estado, FPS y estado Modbus."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.log_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        if self.log_path.stat().st_size == 0:
            self.writer.writerow(
                [
                    "timestamp",
                    "hojas",
                    "palos",
                    "palo_pct_raw",
                    "palo_pct_filtrado",
                    "estado_calidad",
                    "fps",
                    "modbus_ok",
                ]
            )
            self.file.flush()

    def write(
        self,
        hojas: int,
        palos: int,
        quality: QualityResult,
        fps: float,
        modbus_ok: bool,
    ) -> None:
        """Agrega una fila por ciclo procesado."""
        timestamp = datetime.now().isoformat(timespec="seconds")
        self.writer.writerow(
            [
                timestamp,
                hojas,
                palos,
                f"{quality.raw_palo_pct:.2f}",
                f"{quality.filtered_palo_pct:.2f}",
                quality.status_label,
                f"{fps:.2f}",
                int(modbus_ok),
            ]
        )
        self.file.flush()

    def close(self) -> None:
        """Cierra el archivo CSV."""
        self.file.close()


def parse_args() -> AutomationConfig:
    """Argumentos CLI para puesta en marcha y ajuste en planta."""
    parser = argparse.ArgumentParser(
        description="Sprint 3 - Vision artificial + control de calidad + Modbus TCP."
    )
    parser.add_argument("--model", type=Path, default=Path("best.onnx"), help="Ruta a best.onnx")
    parser.add_argument("--camera", type=int, default=0, help="Indice de camara OpenCV")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de entrada ONNX")
    parser.add_argument("--conf", type=float, default=0.35, help="Umbral de confianza")
    parser.add_argument("--iou", type=float, default=0.45, help="Umbral IoU para NMS")
    parser.add_argument("--mask-thres", type=float, default=0.50, help="Umbral binario de mascara")
    parser.add_argument("--max-det", type=int, default=100, help="Maximo de detecciones por frame")
    parser.add_argument("--cam-width", type=int, default=640, help="Ancho solicitado a la camara")
    parser.add_argument("--cam-height", type=int, default=480, help="Alto solicitado a la camara")
    parser.add_argument("--cam-fps", type=int, default=15, help="FPS solicitado a la camara")
    parser.add_argument("--window", type=int, default=10, help="Ventana de media movil en frames")
    parser.add_argument(
        "--max-palo-pct",
        type=float,
        default=30.0,
        help="Porcentaje maximo permitido de palo antes de RECHAZAR",
    )
    parser.add_argument("--plc-ip", type=str, default="192.168.1.100", help="IP del PLC Siemens")
    parser.add_argument("--plc-port", type=int, default=502, help="Puerto Modbus TCP del PLC")
    parser.add_argument("--plc-unit-id", type=int, default=1, help="Unit ID Modbus")
    parser.add_argument(
        "--plc-reconnect-interval",
        type=float,
        default=3.0,
        help="Segundos minimos entre intentos de reconexion al PLC",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Segundos minimos entre fotos consecutivas de rechazo",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("Docs") / "inferencia_sprint3_log.csv",
        help="CSV de trazabilidad industrial",
    )
    parser.add_argument(
        "--rejection-dir",
        type=Path,
        default=Path("Docs") / "rechazos",
        help="Carpeta de imagenes de auditoria",
    )
    args = parser.parse_args()

    return AutomationConfig(
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
        window=args.window,
        max_palo_pct=args.max_palo_pct,
        plc_ip=args.plc_ip,
        plc_port=args.plc_port,
        plc_unit_id=args.plc_unit_id,
        plc_reconnect_interval=args.plc_reconnect_interval,
        cooldown=args.cooldown,
        log_path=args.log,
        rejection_dir=args.rejection_dir,
    )


def main() -> None:
    """Loop principal: camara, IA, calidad, Modbus, auditoria y parada segura."""
    config = parse_args()
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    camera: Optional[ThreadedCamera] = None
    logger: Optional[AutomationCSVLogger] = None
    plc: Optional[SiemensModbusClient] = None

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
        quality_filter = MovingAverageQualityFilter(
            window=config.window,
            max_palo_pct=config.max_palo_pct,
        )
        plc = SiemensModbusClient(
            plc_ip=config.plc_ip,
            plc_port=config.plc_port,
            unit_id=config.plc_unit_id,
            reconnect_interval=config.plc_reconnect_interval,
        )
        plc.connect(force=True)
        audit_saver = AuditImageSaver(config.rejection_dir, config.cooldown)
        logger = AutomationCSVLogger(config.log_path)

        print("[INFO] Sprint 3 iniciado. Ctrl+C para parada segura.", flush=True)
        frame_number = 0

        while not stop_event.is_set():
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            start_time = time.perf_counter()
            detections = model.infer(frame)
            hojas, palos = count_classes(detections)
            quality = quality_filter.update(hojas, palos)

            modbus_ok = plc.write_quality_registers(
                hojas=hojas,
                palos=palos,
                filtered_pct=quality.filtered_palo_pct,
                status=quality.status,
            )

            saved_path = None
            if quality.status == QUALITY_REJECTED:
                saved_path = audit_saver.save_if_allowed(frame, quality.filtered_palo_pct)

            elapsed = time.perf_counter() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            frame_number += 1

            logger.write(hojas, palos, quality, fps, modbus_ok)

            audit_msg = f" | Auditoria: {saved_path.name}" if saved_path else ""
            print(
                f"[FRAME {frame_number}] Hojas: {hojas} | Palos: {palos} | "
                f"Palo raw: {quality.raw_palo_pct:.2f}% | "
                f"Palo filtrado: {quality.filtered_palo_pct:.2f}% | "
                f"Estado: {quality.status_label} | PLC: {'OK' if modbus_ok else 'SIN_LINK'} | "
                f"FPS: {fps:.1f}{audit_msg}",
                flush=True,
            )

    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
    finally:
        if camera is not None:
            camera.stop()
        if plc is not None:
            # Estado seguro al finalizar: sin detecciones y calidad OK.
            if plc.is_connected():
                plc.write_quality_registers(0, 0, 0.0, QUALITY_OK)
            plc.close()
        if logger is not None:
            logger.close()
        print("[INFO] Parada segura completada. Recursos liberados.", flush=True)


if __name__ == "__main__":
    main()
