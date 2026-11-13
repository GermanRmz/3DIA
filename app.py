"""App web: sube una imagen y descarga su modelo 3D con textura.

Uso:
    python app.py
y abre http://localhost:7860
"""
import tempfile

import gradio as gr

from generar3d import generar


def convertir(imagen, pasos, resolucion, semilla, con_textura, resolucion_textura, max_caras):
    if imagen is None:
        raise gr.Error("Sube una imagen primero.")
    salida = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    try:
        generar(imagen, salida, int(pasos), int(resolucion), int(semilla),
                textura=con_textura, resolucion_textura=int(resolucion_textura),
                max_caras=int(max_caras))
    except RuntimeError as e:
        raise gr.Error(str(e))
    return salida, salida


with gr.Blocks(title="Imagen a 3D") as demo:
    gr.Markdown("# Imagen a 3D\nSube una imagen con fondo liso y genera su modelo 3D con color. "
                "Funciona con [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).")
    with gr.Row():
        with gr.Column():
            entrada = gr.Image(type="pil", image_mode="RGBA", label="Imagen")
            con_textura = gr.Checkbox(value=True, label="Pintar textura (color)")
            boton = gr.Button("Generar 3D", variant="primary")
            with gr.Accordion("Opciones de forma", open=False):
                pasos = gr.Slider(1, 30, value=5, step=1, label="Pasos")
                resolucion = gr.Slider(128, 384, value=256, step=32, label="Resolución de la malla")
                semilla = gr.Number(value=0, precision=0, label="Semilla")
            with gr.Accordion("Opciones de textura", open=False):
                resolucion_textura = gr.Radio([1024, 2048], value=2048, label="Resolución de la textura")
                max_caras = gr.Slider(10000, 100000, value=40000, step=5000,
                                      label="Caras máximas al pintar")
        with gr.Column():
            visor = gr.Model3D(label="Modelo 3D")
            archivo = gr.File(label="Descargar .glb")
    boton.click(convertir,
                [entrada, pasos, resolucion, semilla, con_textura, resolucion_textura, max_caras],
                [visor, archivo])

if __name__ == "__main__":
    demo.queue().launch()
