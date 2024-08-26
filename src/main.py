import os
import cv2
import easyocr
import numpy as np
from collections import Counter
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------- CONFIGURACIONES --------------------

INPUT_VIDEO = './Videos/3_multiple.mp4'  # Archivo de video a procesar, si WEBCAM es False
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

# Inicializa el lector de OCR de EasyOCR
reader = easyocr.Reader(['en'])

# Diccionario para almacenar las trayectorias de los objetos rastreados
trajectories = {}

# -------------------- FUNCIONES --------------------

def initialize_tracker():
    """
    Inicializa y retorna un tracker DeepSORT con los parámetros configurados.
    """
    return DeepSort(max_age=MAX_AGE, 
                    n_init=N_INIT, 
                    embedder="mobilenet", 
                    max_cosine_distance=COSINE_DISTANCE_THRESHOLD, 
                    nn_budget=None)

def process_detections(detections, frame, tracker, ocr_history):
    """
    Procesa las detecciones del frame actual, actualiza el tracker y el historial de OCR.
    También dibuja las trayectorias de los objetos rastreados.
    """
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        handle_track(track, frame, ocr_history)

        # Actualiza y dibuja la trayectoria
        track_id = track.track_id
        ltrb = track.to_ltrb()
        center_x, center_y = int((ltrb[0] + ltrb[2]) / 2), int((ltrb[1] + ltrb[3]) / 2)
        
        # Añadir el centro del objeto a la trayectoria del track_id
        if track_id not in trajectories:
            trajectories[track_id] = []
        
        trajectories[track_id].append((center_x, center_y))
        
        # Limitar la longitud de la trayectoria a TRAJECTORY_LENGTH
        if len(trajectories[track_id]) > TRAJECTORY_LENGTH:
            trajectories[track_id] = trajectories[track_id][-TRAJECTORY_LENGTH:]

        # Dibuja la trayectoria si está habilitado SHOW_TRAJECTORY
        if SHOW_TRAJECTORY:
            draw_trajectory(frame, trajectories[track_id])

def handle_track(track, frame, ocr_history):
    """
    Maneja un objeto rastreado, aplica OCR y actualiza el historial de texto por track_id.
    """
    track_id = track.track_id
    ltrb = track.to_ltrb()  # Obtiene las coordenadas del bounding box rastreado
    x1, y1, x2, y2 = map(int, ltrb)

    # Asegurarse de que las coordenadas estén dentro del frame
    H, W, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    if x2 - x1 > 0 and y2 - y1 > 0:  # Verifica que las coordenadas sean válidas
        img_license = frame[y1:y2, x1:x2]

        # Aplica OCR sobre la imagen recortada
        ocr_results = reader.readtext(img_license)

        if track_id not in ocr_history:
            ocr_history[track_id] = []

        if ocr_results:
            # Asegurarse de que el OCR devuelve resultados válidos
            _, text, confidence = ocr_results[0]
            if isinstance(text, str) and confidence >= OCR_CONFIDENCE_THRESHOLD:
                ocr_history[track_id].append(text)

        # Calcula el valor más frecuente en el historial de OCR
        most_common_plate = get_most_common_plate(ocr_history, track_id)

        # Dibuja la patente detectada y el bounding box en el frame original
        draw_plate_on_frame(frame, most_common_plate, x1, y1, x2, y2, track_id)

def get_most_common_plate(ocr_history, track_id):
    """
    Retorna la patente más frecuente en el historial de OCR para un track_id.
    """
    if ocr_history[track_id]:
        return Counter(ocr_history[track_id]).most_common(1)[0][0]
    return "Plate"

def draw_plate_on_frame(frame, plate_text, x1, y1, x2, y2, track_id=None):
    """
    Dibuja el texto de la patente y el bounding box en el frame.
    También añade un fondo negro detrás del texto.
    Si SHOW_TRACKER_ID está habilitado, muestra el ID del tracker.
    """
    # Configuraciones del texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text = plate_text.upper()

    # Agregar el ID del tracker al texto si está habilitado SHOW_TRACKER_ID
    if SHOW_TRACKER_ID and track_id is not None:
        text = f'ID {track_id}: {text}'

    # Obtener el tamaño del texto y la posición
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x, text_y = x1, y1 - 10  # Posición del texto (arriba del bounding box)
    box_x1, box_y1 = text_x, text_y - text_size[1]
    box_x2, box_y2 = text_x + text_size[0], text_y

    # Dibujar un rectángulo negro detrás del texto
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)

    # Dibuja el texto encima del rectángulo negro
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

    # Dibujar el bounding box alrededor de la patente
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def draw_trajectory(frame, trajectory):
    """
    Dibuja la trayectoria de un objeto rastreado en el frame.
    Conecta los puntos que representan el centro del objeto.
    El punto actual es verde, y los siguientes puntos son más claros hasta llegar al blanco.
    """
    trajectory_length = len(trajectory)
    for i in range(1, trajectory_length):
        if trajectory[i - 1] is None or trajectory[i] is None:
            continue

        # Calcular el color del punto actual (verde a blanco)
        ratio = i / trajectory_length
        color = (
            int(255 * ratio),  # Rojo (aumenta hacia el blanco)
            255,               # Verde (permanece constante)
            int(255 * ratio)    # Azul (aumenta hacia el blanco)
        )

        # Dibuja una línea entre los puntos sucesivos en la trayectoria
        cv2.line(frame, trajectory[i - 1], trajectory[i], color, 2)

def setup_video_writer(input_video_path, frame_width, frame_height, fps):
    """
    Configura y retorna un VideoWriter para almacenar el video procesado.
    """
    base_name, ext = os.path.splitext(input_video_path)
    output_video_path = f"{base_name}_processed{ext}"
    return cv2.VideoWriter(output_video_path, 
                           cv2.VideoWriter_fourcc(*'mp4v'), 
                           fps, (frame_width, frame_height)), output_video_path

# -------------------- LÓGICA PRINCIPAL --------------------

def main():
    # Configura la fuente de video (webcam o archivo de video)
    if WEBCAM:
        cap = cv2.VideoCapture(0)  # Usa la webcam por defecto
        video_writer = None  # No guardamos el video procesado si es webcam
    else:
        video_path = INPUT_VIDEO  # Usa un archivo de video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("No se pudo abrir la fuente de video.")
            return

        # Obtener los detalles del video original
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Configurar el VideoWriter si SAVE_TO_VIDEO es True
        if SAVE_TO_VIDEO:
            video_writer, output_video_path = setup_video_writer(video_path, frame_width, frame_height, fps)
        else:
            video_writer = None

    # Carga el modelo de detección YOLO y el tracker
    model = YOLO(MODEL_PATH)
    tracker = initialize_tracker()

    # Diccionario para almacenar los valores de OCR por ID
    ocr_history = {}

    # Convertimos la tecla en un código ASCII (para manejar diferentes teclas de cierre)
    close_key = ord(CLOSE_WINDOW_KEY.lower())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Obtener los resultados de detección con YOLO
        results = model(frame)[0]

        # Convertir detecciones al formato esperado por DeepSORT
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > YOLO_THRESHOLD:
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detections.append((bbox, score, class_id))

        # Procesar las detecciones
        if detections:
            process_detections(detections, frame, tracker, ocr_history)

        # Guardar el frame en el archivo de video procesado si SAVE_TO_VIDEO está habilitado
        if video_writer is not None:
            video_writer.write(frame)

        # Mostrar el frame procesado en tiempo real si DISPLAY_VIDEO está activado
        if DISPLAY_VIDEO:
            cv2.imshow("Salida Procesada", frame)

            # Presionar la tecla configurada para salir
            if cv2.waitKey(1) & 0xFF == close_key:
                break

    cap.release()

    # Liberar el VideoWriter si se guardó el video
    if video_writer is not None:
        video_writer.release()
        print(f"Video procesado guardado en: {output_video_path}")

    if DISPLAY_VIDEO:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
