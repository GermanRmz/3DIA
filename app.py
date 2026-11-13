"""App web: sube una imagen y descarga su modelo 3D con textura.

Uso:
    python app.py
y abre http://localhost:7860
"""
import os
import tempfile
import time

import gradio as gr

import acabados
import segmentar
from generar3d import generar, ruta_color, ruta_impresion, ruta_partes
from preparar_imagen import preparar


def convertir(imagen, fondo, ajustar, pasos, resolucion, semilla, con_textura, resolucion_textura,
              max_caras, con_stl, acabado, altura_mm, escala, profundidad, colores, intentos,
              seg_modo, seg_partes, seg_volumen, seg_cortes, seg_union, seg_diametro, seg_largo,
              seg_holgura, seg_iman_d, seg_iman_h, progreso=gr.Progress(track_tqdm=True)):
    """Genera paso a paso y va mostrando el avance en la página."""
    if imagen is None:
        raise gr.Error("Sube una imagen primero.")
    inicio = time.time()
    salida = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    try:
        progreso(0, desc="Preparando la imagen...")
        preparada, notas = preparar(imagen, FONDOS[fondo], ajustar)
        estado = "\n".join(f"- {n}" for n in notas)
        pasos_txt = "forma y textura (unos minutos)" if con_textura else "forma"
        yield preparada, None, None, None, None, None, estado + f"\n\n**Generando {pasos_txt}...**"

        segmentacion = None
        modo = MODOS_UI[seg_modo]
        if con_stl and modo != "no":
            try:
                volumen = tuple(float(v) for v in seg_volumen.lower().replace(" ", "").split("x"))
                assert len(volumen) == 3
            except (ValueError, AssertionError):
                raise gr.Error("Volumen de impresión no válido. Ejemplo: 220x220x250")
            segmentacion = {"modo": modo, "partes": int(seg_partes), "volumen": volumen,
                            "manual": seg_cortes,
                            "union": segmentar.OpcionesUnion(UNIONES_UI[seg_union], float(seg_diametro),
                                                             float(seg_largo), float(seg_holgura),
                                                             float(seg_iman_d), float(seg_iman_h))}

        generar(preparada, salida, int(pasos), int(resolucion), int(semilla),
                textura=con_textura, fondo="mantener", ajustar=False, intentos=int(intentos),
                colores=int(colores), acabado=ACABADOS_UI[acabado] if con_stl else None, altura_mm=float(altura_mm),
                escala_acabado=float(escala), profundidad_acabado=float(profundidad),
                segmentacion=segmentacion,
                resolucion_textura=int(resolucion_textura), max_caras=int(max_caras))
    except (RuntimeError, ValueError) as e:
        raise gr.Error(str(e))
    descargas, vista = None, None
    if con_stl:
        descargas = [ruta_impresion(salida)]
        vista = ruta_impresion(salida)
        if con_textura and int(colores) > 1:
            descargas.append(ruta_color(salida, ".3mf"))
            vista = ruta_color(salida, ".glb")
    vista_partes = None
    if segmentacion and os.path.exists(ruta_partes(salida, ".zip")):
        descargas += [ruta_partes(salida, ".zip"), ruta_partes(salida, ".3mf")]
        vista_partes = ruta_partes(salida, ".glb")
    yield (preparada, salida, salida, vista, descargas, vista_partes,
           estado + f"\n\n**Listo en {time.time() - inicio:.0f} s.**")


def visibilidad_segmentacion(modo, union):
    m, u = MODOS_UI[modo], UNIONES_UI[union]
    activo = m != "no"
    return (gr.update(visible=m in ("horizontal", "vertical_x", "vertical_y")),
            gr.update(visible=m == "cama"), gr.update(visible=m == "manual"),
            gr.update(visible=activo),
            gr.update(visible=activo and u in ("espiga", "pasadores")),
            gr.update(visible=activo and u == "imanes"))


def valores_acabado(nombre):
    """Escala y profundidad recomendadas al cambiar de acabado."""
    clave = ACABADOS_UI[nombre]
    if clave == "ninguno":
        return gr.update(interactive=False), gr.update(interactive=False)
    _, escala, profundidad, _ = acabados.ACABADOS[clave]
    return gr.update(value=escala, interactive=True), gr.update(value=profundidad, interactive=True)


MODOS_UI = {"No dividir": "no", **{v: k for k, v in segmentar.MODOS.items()}}
UNIONES_UI = {v: k for k, v in segmentar.UNIONES.items()}

ACABADOS_UI = {"Sin acabado (liso)": "ninguno",
               **{f"{k.capitalize()}: {v[3].lower()}": k for k, v in acabados.ACABADOS.items()}}

FONDOS = {"Automático": "auto", "IA": "ia", "Fondo blanco (sin IA)": "blanco",
          "No tocar": "mantener"}

with gr.Blocks(title="Imagen a 3D") as demo:
    gr.Markdown("# Imagen a 3D\nSube una foto de un objeto y genera su modelo 3D con color y un STL listo para imprimir. "
                "Funciona con [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).")
    with gr.Row():
        with gr.Column():
            entrada = gr.Image(type="pil", image_mode="RGBA", label="Imagen")
            fondo = gr.Radio(list(FONDOS), value="Automático", label="Quitar el fondo")
            ajustar = gr.Checkbox(value=True, label="Ajustar imagen (recortar, centrar, limpiar)")
            con_textura = gr.Checkbox(value=True, label="Pintar textura (color)")
            with gr.Accordion("Impresión 3D (.stl)", open=True):
                con_stl = gr.Checkbox(value=True, label="Generar STL para imprimir")
                acabado = gr.Dropdown(list(ACABADOS_UI), value="Sin acabado (liso)",
                                      label="Acabado de la superficie (relieve real)")
                altura_mm = gr.Slider(20, 250, value=60, step=5, label="Altura de la pieza (mm)")
                colores = gr.Slider(1, 8, value=1, step=1,
                                    label="Colores de filamento (impresora multicolor, necesita "
                                          "textura; 1 = un solo color)")
                with gr.Row():
                    escala = gr.Number(value=2.5, label="Tamaño del patrón (mm)", interactive=False)
                    profundidad = gr.Number(value=0.5, label="Profundidad (mm)", interactive=False)
            with gr.Accordion("Dividir en partes (piezas grandes)", open=False):
                seg_modo = gr.Dropdown(list(MODOS_UI), value="No dividir", label="Cómo dividir")
                seg_partes = gr.Slider(2, 8, value=2, step=1, label="Número de partes", visible=False)
                seg_volumen = gr.Textbox(value="220x220x250", label="Volumen de tu impresora (mm, AnchoxFondoxAlto)",
                                         visible=False)
                seg_cortes = gr.Textbox(value="z=30", label="Cortes (mm): eje=posición separados por comas",
                                        visible=False)
                with gr.Group(visible=False) as grupo_union:
                    seg_union = gr.Radio(list(UNIONES_UI), value=segmentar.UNIONES["espiga"],
                                         label="Tipo de unión")
                    with gr.Row(visible=True) as fila_espiga:
                        seg_diametro = gr.Number(value=5.0, label="Diámetro (mm)")
                        seg_largo = gr.Number(value=8.0, label="Largo (mm)")
                        seg_holgura = gr.Number(value=0.2, label="Holgura (mm)")
                    with gr.Row(visible=False) as fila_iman:
                        seg_iman_d = gr.Number(value=6.0, label="Diámetro del imán (mm)")
                        seg_iman_h = gr.Number(value=3.0, label="Alto del imán (mm)")
            boton = gr.Button("Generar 3D", variant="primary")
            with gr.Accordion("Opciones de forma", open=False):
                pasos = gr.Slider(1, 30, value=5, step=1, label="Pasos")
                resolucion = gr.Slider(128, 384, value=256, step=32, label="Resolución de la malla")
                semilla = gr.Number(value=0, precision=0, label="Semilla")
                intentos = gr.Slider(1, 10, value=6, step=1,
                                     label="Intentos si el modelo sale plano (otras semillas)")
            with gr.Accordion("Opciones de textura", open=False):
                resolucion_textura = gr.Radio([1024, 2048], value=2048, label="Resolución de la textura")
                max_caras = gr.Slider(10000, 100000, value=40000, step=5000,
                                      label="Caras máximas al pintar")
        with gr.Column():
            estado = gr.Markdown()
            preparada = gr.Image(label="Imagen preparada (lo que recibe el modelo)", height=256)
            visor = gr.Model3D(label="Modelo 3D")
            archivo = gr.File(label="Descargar .glb (con color)")
            visor_stl = gr.Model3D(label="Modelo para imprimir (mm, con acabado y colores)")
            visor_partes = gr.Model3D(label="Partes (vista separada)")
            archivo_stl = gr.File(label="Descargar para imprimir (.stl, .3mf, partes .zip)",
                                  file_count="multiple")
    boton.click(convertir,
                [entrada, fondo, ajustar, pasos, resolucion, semilla, con_textura, resolucion_textura,
                 max_caras, con_stl, acabado, altura_mm, escala, profundidad, colores, intentos,
                 seg_modo, seg_partes, seg_volumen, seg_cortes, seg_union, seg_diametro, seg_largo,
                 seg_holgura, seg_iman_d, seg_iman_h],
                [preparada, visor, archivo, visor_stl, archivo_stl, visor_partes, estado],
                show_progress_on=[visor])
    acabado.change(valores_acabado, acabado, [escala, profundidad])
    for control in (seg_modo, seg_union):
        control.change(visibilidad_segmentacion, [seg_modo, seg_union],
                       [seg_partes, seg_volumen, seg_cortes, grupo_union, fila_espiga, fila_iman])

if __name__ == "__main__":
    demo.queue().launch()
