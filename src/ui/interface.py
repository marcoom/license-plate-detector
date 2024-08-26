# src/ui/interface.py
import os
import cv2
from config import SHOW_TRACKER_ID

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