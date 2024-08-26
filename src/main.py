# src/main.py
import easyocr
import cv2
from config import WEBCAM, INPUT_VIDEO, SAVE_TO_VIDEO, MODEL_PATH, CLOSE_WINDOW_KEY, YOLO_THRESHOLD, DISPLAY_VIDEO
from ui.interface import setup_video_writer
from tracking.tracker import initialize_tracker, process_detections
from ultralytics import YOLO

# # Inicializa el lector de OCR de EasyOCR
# reader = easyocr.Reader(['en'])

# # Diccionario para almacenar las trayectorias de los objetos rastreados
# trajectories = {}


def main():
    # Inicializa el lector de OCR de EasyOCR
    reader = easyocr.Reader(['en'])

    # Diccionario para almacenar las trayectorias de los objetos rastreados
    trajectories = {}

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
            process_detections(detections, frame, tracker, ocr_history, trajectories, reader)

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
