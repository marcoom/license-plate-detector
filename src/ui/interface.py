# src/ui/interface.py
import gradio as gr
from src.main import process_video

def video_processing(video_path):
    return process_video(video_path)

iface = gr.Interface(
    fn=video_processing,
    inputs=gr.inputs.Video(type='filepath'),
    outputs=gr.outputs.Video(type='numpy', fps=20),
    live=True,
    description="Carga un video para detectar y rastrear patentes de autos."
)

if __name__ == "__main__":
    iface.launch(debug=True)