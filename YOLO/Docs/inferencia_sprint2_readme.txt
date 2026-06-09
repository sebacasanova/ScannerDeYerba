DOCUMENTACION SPRINT 2 - INFERENCIA Y CONTEO EN TIEMPO REAL

Objetivo
--------
Ejecutar en Raspberry Pi 3B+ un modelo YOLOv8 Nano Segmentacion exportado a ONNX
para contar en tiempo real:

0: hoja
1: palo

El script esta preparado para entorno headless por SSH. No usa cv2.imshow(),
no abre ventanas y no importa la libreria ultralytics.


Archivo principal
-----------------
inferencia_yerba_sprint2.py


Backend de inferencia
---------------------
Se utiliza onnxruntime con CPUExecutionProvider:

- Menor carga que importar ultralytics.
- Compatible con CPU/ARM.
- Adecuado para Raspberry Pi 3B+.

El script implementa manualmente el postprocesamiento de YOLOv8 Segmentacion:

1. Preprocesamiento letterbox a 640x640.
2. Inferencia ONNX.
3. Extraccion de cajas en formato xywh.
4. Extraccion de scores por clase.
5. Seleccion de clase con mayor confianza.
6. NMS por clase.
7. Combinacion de coeficientes de mascara con prototipos proto.
8. Aplicacion de sigmoid.
9. Reescalado de mascaras al frame original.
10. Binarizacion de mascaras.
11. Extraccion de contornos con cv2.findContours.
12. Conteo de hojas y palos.


Captura de camara
-----------------
La camara se captura en un hilo secundario usando threading.

La clase ThreadedCamera mantiene siempre el ultimo frame disponible, evitando
que el procesamiento de la IA bloquee la lectura de la camara. Esto ayuda a
maximizar los FPS efectivos en hardware limitado.


Ejecucion por SSH
-----------------
Copiar best.onnx a la misma carpeta del script o indicar la ruta con --model.

Comando basico:

python3 inferencia_yerba_sprint2.py --model best.onnx

Comando con parametros explicitos:

python3 inferencia_yerba_sprint2.py \
  --model best.onnx \
  --camera 0 \
  --imgsz 640 \
  --conf 0.35 \
  --iou 0.45 \
  --cam-width 640 \
  --cam-height 480 \
  --cam-fps 15


Salida por consola
------------------
Por cada frame procesado se imprime una linea como:

[FRAME 25] Hojas detectadas: 14 | Palos detectados: 3 | FPS: 4.8


Registro persistente
--------------------
El script crea automaticamente la carpeta Docs si no existe.

Luego genera o actualiza:

Docs/inferencia_sprint2_log.csv

Columnas:

- timestamp
- hojas
- palos
- fps


Recomendaciones para Raspberry Pi 3B+
-------------------------------------
- Instalar onnxruntime para ARM si esta disponible en la distribucion usada.
- Mantener resolucion de camara en 640x480 para reducir carga.
- Si los FPS son bajos, probar:
  - --imgsz 512 si se exporta un modelo compatible a ese tamano.
  - --conf 0.40 para reducir detecciones candidatas.
  - --max-det 50 si hay demasiadas instancias.
- Evitar escritorio grafico durante la inferencia.
- Cerrar otros procesos consumidores de RAM.


Dependencias recomendadas
-------------------------
pip3 install numpy opencv-python onnxruntime

En Raspberry Pi puede convenir instalar OpenCV desde paquetes del sistema:

sudo apt install python3-opencv

Luego instalar solo:

pip3 install numpy onnxruntime

