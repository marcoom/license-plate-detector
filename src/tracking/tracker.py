# src/tracking/tracker.py
from src.config import COSINE_DISTANCE_THRESHOLD, MAX_AGE, N_INIT, SHOW_TRAJECTORY, TRAJECTORY_LENGTH, OCR_CONFIDENCE_THRESHOLD
from deep_sort_realtime.deepsort_tracker import DeepSort
#from main import trajectories, reader
from ui.interface import draw_trajectory, draw_plate_on_frame
from ocr.ocr import get_most_common_plate

def initialize_tracker():
    """
    Inicializa y retorna un tracker DeepSORT con los parámetros configurados.
    """
    return DeepSort(max_age=MAX_AGE, 
                    n_init=N_INIT, 
                    embedder="mobilenet", 
                    max_cosine_distance=COSINE_DISTANCE_THRESHOLD, 
                    nn_budget=None)

def process_detections(detections, frame, tracker, ocr_history, trajectories, reader):
    """
    Procesa las detecciones del frame actual, actualiza el tracker y el historial de OCR.
    También dibuja las trayectorias de los objetos rastreados.
    """
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        handle_track(track, frame, ocr_history, reader)

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

def handle_track(track, frame, ocr_history, reader):
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