"""App web: sube una imagen y descarga su modelo 3D.

Uso:
    python app.py
y abre http://localhost:7860
"""
import tempfile

import gradio as gr

from generar3d import generar


def convertir(imagen, pasos, resolucion, semilla):
    if imagen is None:
        raise gr.Error("Sube una imagen primero.")
    salida = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    generar(imagen, salida, int(pasos), int(resolucion), int(semilla))
    return salida, salida


with gr.Blocks(title="Imagen a 3D") as demo:
    gr.Markdown("# Imagen a 3D\nSube una imagen con fondo liso y genera su modelo 3D. "
                "Funciona con [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).")
    with gr.Row():
        with gr.Column():
            entrada = gr.Image(type="pil", image_mode="RGBA", label="Imagen")
            pasos = gr.Slider(1, 30, value=5, step=1, label="Pasos")
            resolucion = gr.Slider(128, 384, value=256, step=32, label="Resolución de la malla")
            semilla = gr.Number(value=0, precision=0, label="Semilla")
            boton = gr.Button("Generar 3D", variant="primary")
        with gr.Column():
            visor = gr.Model3D(label="Modelo 3D")
            archivo = gr.File(label="Descargar .glb")
    boton.click(convertir, [entrada, pasos, resolucion, semilla], [visor, archivo])

if __name__ == "__main__":
    demo.launch()
