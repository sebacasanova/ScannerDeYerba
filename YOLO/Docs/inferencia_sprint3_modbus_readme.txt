DOCUMENTACION SPRINT 3 - AUTOMATIZACION, CALIDAD Y MODBUS TCP

Objetivo
--------
Integrar la inferencia ONNX del Sprint 2 con logica de control de calidad,
comunicacion industrial Modbus TCP hacia PLC Siemens S7-1200/1500 y guardado de
imagenes de auditoria ante muestras rechazadas.


Archivo principal
-----------------
inferencia_yerba_sprint3_modbus.py

Este archivo reutiliza:

- ThreadedCamera
- YOLOv8SegONNX
- count_classes

desde inferencia_yerba_sprint2.py.


Registros Modbus TCP
--------------------
El PLC Siemens debe actuar como servidor Modbus TCP. La Raspberry Pi actua como
cliente y escribe Holding Registers desde la direccion 0:

HR0: Cantidad de hojas del frame actual
HR1: Cantidad de palos del frame actual
HR2: Porcentaje de palo filtrado * 100
HR3: Estado de calidad, 0 OK, 1 RECHAZADO

Ejemplo:
25.45% se envia en HR2 como 2545.


Control de calidad
------------------
Porcentaje crudo por frame:

(palos / (hojas + palos)) * 100

Luego se aplica media movil con collections.deque. La ventana se configura con:

--window

Valor por defecto:

10 frames

El limite de rechazo se configura con:

--max-palo-pct

Valor por defecto:

30.0

Si el porcentaje filtrado supera el limite, el estado de calidad pasa a:

RECHAZADA


Imagenes de auditoria
---------------------
Cuando la muestra queda RECHAZADA, el script guarda una imagen en:

Docs/rechazos/

El archivo incluye timestamp y porcentaje de palo filtrado:

rechazo_20260609_200000_palo_35.00pct.jpg

Para evitar rafagas de imagenes repetidas, se usa cooldown:

--cooldown

Valor por defecto:

2.0 segundos


Ejecucion basica en Raspberry Pi por SSH
----------------------------------------
python3 inferencia_yerba_sprint3_modbus.py --model best.onnx


Ejecucion con parametros industriales
-------------------------------------
python3 inferencia_yerba_sprint3_modbus.py \
  --model best.onnx \
  --camera 0 \
  --plc-ip 192.168.1.100 \
  --plc-port 502 \
  --window 10 \
  --max-palo-pct 30.0 \
  --cooldown 2.0


Dependencias
------------
sudo apt install python3-opencv
pip3 install numpy onnxruntime pymodbus


Notas de puesta en marcha
-------------------------
- Confirmar conectividad con el PLC: ping 192.168.1.100
- Confirmar que el servidor Modbus TCP este habilitado en el PLC.
- Validar que el area de Holding Registers usada por Siemens este mapeada desde
  direccion 0.
- Ajustar --max-palo-pct en planta con muestras reales aceptadas y rechazadas.
- Si el enlace cae, el script intenta reconectar automaticamente.

