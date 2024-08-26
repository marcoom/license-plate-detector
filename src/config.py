# src/config.py

# Video y procesamiento
INPUT_VIDEO = './data/3_multiple.mp4'  # Archivo de video a procesar
WEBCAM = True  # Si es True, usa la webcam en lugar del archivo de video
DISPLAY_VIDEO = True  # Si es True, muestra el video procesado en tiempo real
SAVE_TO_VIDEO = True  # Si es True, guarda el video procesado cuando WEBCAM=False
SHOW_TRACKER_ID = True  # Si es True, muestra el ID del tracker junto con la patente
SHOW_TRAJECTORY = True  # Si es True, muestra la trayectoria de los objetos rastreados
CLOSE_WINDOW_KEY = 'q'  # Tecla para cerrar la ventana de video

# Modelo y parámetros de procesamiento
MODEL_PATH = './models/car_plate.pt'
YOLO_THRESHOLD = 0.5  # Umbral para detección de objetos
OCR_CONFIDENCE_THRESHOLD = 0.02  # Umbral de confianza mínimo para resultados OCR
MAX_AGE = 60  # Número máximo de frames que el objeto puede estar ausente antes de eliminar el track
N_INIT = 10  # Número de detecciones necesarias para confirmar un objeto
COSINE_DISTANCE_THRESHOLD = 0.4  # Umbral de distancia de similitud
TRAJECTORY_LENGTH = 50  # Longitud máxima de la trayectoria
