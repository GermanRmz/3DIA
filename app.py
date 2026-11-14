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
import mecanismos
import personaje
import segmentar
from dividir import cargar, dividir_modelo
from generar3d import generar, ruta_color, ruta_impresion
from preparar_imagen import preparar


def convertir(imagen, fondo, ajustar, pasos, resolucion, semilla, con_textura, resolucion_textura,
              max_caras, con_stl, acabado, altura_mm, escala, profundidad, colores, intentos,
              progreso=gr.Progress(track_tqdm=True)):
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

        generar(preparada, salida, int(pasos), int(resolucion), int(semilla),
                textura=con_textura, fondo="mantener", ajustar=False, intentos=int(intentos),
                colores=int(colores), acabado=ACABADOS_UI[acabado] if con_stl else None, altura_mm=float(altura_mm),
                escala_acabado=float(escala), profundidad_acabado=float(profundidad),
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
    liberar_si_falta_ram()
    # El .glb con color sirve después en la pestaña "Dividir modelo".
    yield (preparada, salida, salida, vista, descargas, salida,
           estado + f"\n\n**Listo en {time.time() - inicio:.0f} s.** Para imprimirlo por piezas, "
                    "usa la pestaña *Dividir modelo*.")


def dividir_app(archivo, ultimo, tipo, plantilla, max_partes, cortes, union_p, union_c, diametro,
                holgura, altura, colores):
    ruta = archivo
    if not ruta and ultimo:
        # La pieza de impresión (con acabado, en mm) y, por su nombre, el .glb con color.
        ruta = ruta_impresion(ultimo) if os.path.exists(ruta_impresion(ultimo)) else ultimo
    if not ruta:
        raise gr.Error("Sube un modelo o genera uno en la pestaña Generar.")
    base = tempfile.NamedTemporaryFile(suffix="", delete=False).name
    if tipo == TIPOS[0]:
        modo, union = PLANTILLAS_UI[plantilla], UNIONES_PERSONAJE[union_p]
    elif tipo == TIPOS[1]:
        modo, union = "auto", UNIONES_PERSONAJE[union_p]
    else:
        modo, union = None, segmentar_uniones[union_c]
    try:
        rutas, informe = dividir_modelo(
            ruta, modo, cortes, union, float(diametro or 0), float(holgura), float(altura) or None,
            int(colores), salida_base=base, max_partes=int(max_partes))
    except (RuntimeError, ValueError) as e:
        raise gr.Error(str(e))
    texto = "\n".join(f"- {linea}" for linea in informe.splitlines())
    return texto, rutas["vista"], [rutas["zip"], rutas["3mf"]]


def _modelo_mecanismo(archivo, ultimo, altura):
    ruta = archivo
    if not ruta and ultimo:
        ruta = ruta_impresion(ultimo) if os.path.exists(ruta_impresion(ultimo)) else ultimo
    if not ruta:
        raise gr.Error("Sube un modelo o genera uno en la pestaña Generar.")
    return cargar(ruta, float(altura) or None)


def analizar_app(archivo, ultimo, imagen, descripcion, altura):
    malla, _ = _modelo_mecanismo(archivo, ultimo, altura)
    try:
        piezas, resumen = mecanismos.analizar(malla, imagen, descripcion or "")
    except Exception as e:  # sin clave, sin red, rechazo...
        raise gr.Error(f"No se pudo analizar con la IA: {e}")
    return mecanismos.a_json(piezas, resumen), mecanismos.vista_previa(malla, piezas), \
        f"**IA:** {resumen}\n\nRevisa o corrige las piezas (JSON) y pulsa *Construir*."


def vista_app(archivo, ultimo, texto, altura):
    malla, _ = _modelo_mecanismo(archivo, ultimo, altura)
    try:
        piezas = mecanismos.desde_json(texto or "{}")
    except (ValueError, TypeError) as e:
        raise gr.Error(f"JSON no válido: {e}")
    return mecanismos.vista_previa(malla, piezas)


def construir_app(archivo, ultimo, texto, modo, holgura, diametro, altura, colores):
    malla, textura = _modelo_mecanismo(archivo, ultimo, altura)
    try:
        piezas = mecanismos.desde_json(texto or "{}")
    except (ValueError, TypeError) as e:
        raise gr.Error(f"JSON no válido: {e}")
    if not piezas:
        raise gr.Error("No hay piezas móviles: pulsa *Analizar con IA* o escríbelas en el JSON.")
    clave = MODOS_MEC[modo]
    try:
        mec = mecanismos.construir(malla, piezas, clave, float(holgura), float(diametro or 0))
    except (RuntimeError, ValueError) as e:
        raise gr.Error(str(e))
    mecanismos.colorear(mec, malla, textura)
    if textura is None:
        mec.avisos.append("El modelo no tiene textura: se usa un color por pieza (sube el .glb con color).")
    base = tempfile.NamedTemporaryFile(suffix="", delete=False).name
    rutas = mecanismos.exportar(mec, base, clave, int(colores))
    lineas = [f"- **{p.nombre}**: {p.tipo}" for p in mec.partes] + [f"- Aviso: {a}" for a in mec.avisos]
    return "\n".join(lineas), rutas["glb"], [rutas["glb"], rutas["zip"], rutas["3mf"]]


def visibilidad_division(tipo):
    personaje_, auto = tipo == TIPOS[0], tipo == TIPOS[1]
    return (gr.update(visible=personaje_), gr.update(visible=personaje_), gr.update(visible=auto),
            gr.update(visible=tipo == TIPOS[2]), gr.update(visible=personaje_ or auto),
            gr.update(visible=tipo == TIPOS[2]))


def liberar_si_falta_ram(minimo_gb=4.0):
    """Si queda poca RAM libre, descarga los modelos que más ocupan (se recargan
    solos al volver a usarlos). Evita que el sistema cierre la app."""
    try:
        with open("/proc/meminfo") as f:
            libre = {l.split(":")[0]: int(l.split()[1]) for l in f}["MemAvailable"] / 1024 ** 2
    except (OSError, KeyError, ValueError):
        return
    if libre < minimo_gb:
        import preparar_imagen
        import textura
        textura.liberar()
        preparar_imagen.liberar()
        print(f"Poca RAM libre ({libre:.1f} GB): se descargan los modelos de pintura y de fondo.")


def valores_acabado(nombre):
    """Escala y profundidad recomendadas al cambiar de acabado."""
    clave = ACABADOS_UI[nombre]
    if clave == "ninguno":
        return gr.update(interactive=False), gr.update(interactive=False)
    _, escala, profundidad, _ = acabados.ACABADOS[clave]
    return gr.update(value=escala, interactive=True), gr.update(value=profundidad, interactive=True)


TIPOS = ["Personaje (cabeza, torso, brazos, piernas)", "Automático (cualquier forma)", "Cortes planos"]
PLANTILLAS_UI = {v[0]: k for k, v in personaje.PLANTILLAS.items()}
UNIONES_PERSONAJE = {v: k for k, v in personaje.UNIONES.items()}
segmentar_uniones = {v: k for k, v in segmentar.UNIONES.items()}
PLANTILLAS_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "plantillas.png")

ACABADOS_UI = {"Sin acabado (liso)": "ninguno",
               **{f"{k.capitalize()}: {v[3].lower()}": k for k, v in acabados.ACABADOS.items()}}

MODOS_MEC = {"Montada (se imprime ya ensamblada, print-in-place)": "montada",
             "Separadas (piezas y pasadores sueltos)": "separadas"}

FONDOS = {"Automático": "auto", "IA": "ia", "Fondo blanco (sin IA)": "blanco",
          "No tocar": "mantener"}

with gr.Blocks(title="Imagen a 3D") as demo:
    gr.Markdown("# Imagen a 3D\nSube una foto de un objeto y genera su modelo 3D con color y un STL listo para imprimir. "
                "Funciona con [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).")
    ultimo = gr.State(None)
    with gr.Tab("Generar"), gr.Row():
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
            archivo_stl = gr.File(label="Descargar para imprimir (.stl, .3mf a color)",
                                  file_count="multiple")
    boton.click(convertir,
                [entrada, fondo, ajustar, pasos, resolucion, semilla, con_textura, resolucion_textura,
                 max_caras, con_stl, acabado, altura_mm, escala, profundidad, colores, intentos],
                [preparada, visor, archivo, visor_stl, archivo_stl, ultimo, estado],
                show_progress_on=[visor])
    acabado.change(valores_acabado, acabado, [escala, profundidad])

    with gr.Tab("Dividir modelo"), gr.Row():
        with gr.Column():
            modelo = gr.File(label="Modelo (.glb de esta app, .stl, .obj). Si lo dejas vacío se usa "
                                   "el último generado", file_types=[".glb", ".gltf", ".stl", ".obj", ".ply"],
                             type="filepath")
            tipo = gr.Radio(TIPOS, value=TIPOS[0], label="Tipo de división")
            imagen_plantillas = gr.Image(PLANTILLAS_IMG, show_label=False, interactive=False,
                                         height=150)
            plantilla = gr.Radio(list(PLANTILLAS_UI), value=list(PLANTILLAS_UI)[0], label="Plantilla",
                                 info="El personaje debe estar de pie, de frente y con los brazos "
                                      "separados del cuerpo para poder separarlos.")
            max_partes = gr.Slider(2, 12, value=8, step=1, label="Máximo de partes", visible=False,
                                   info="Separa lo que sobresale del cuerpo: patas, cola, cabeza, "
                                        "alas, asas...")
            cortes = gr.Textbox(value="z=40", label="Cortes (mm): eje=posición separados por comas",
                                visible=False)
            union_p = gr.Radio(list(UNIONES_PERSONAJE), value=list(UNIONES_PERSONAJE)[0],
                               label="Unión")
            union_c = gr.Radio(list(segmentar_uniones), value=segmentar.UNIONES["espiga"], label="Unión",
                               visible=False)
            with gr.Row():
                diametro = gr.Number(value=0, label="Diámetro de la unión (mm, 0 = automático)")
                holgura = gr.Number(value=0.2, label="Holgura (mm)")
            with gr.Row():
                altura = gr.Number(value=0, label="Altura de la figura (mm, 0 = la del modelo; "
                                                  "un .glb se escala a 60)")
                colores_d = gr.Slider(1, 8, value=1, step=1, label="Colores (necesita .glb con textura)")
            boton_dividir = gr.Button("Dividir", variant="primary")
        with gr.Column():
            informe = gr.Markdown()
            vista = gr.Model3D(label="Partes (vista separada)")
            archivos = gr.File(label="Descargar partes (.zip con un STL por parte, .3mf colocado)",
                               file_count="multiple")
    tipo.change(visibilidad_division, tipo,
                [imagen_plantillas, plantilla, max_partes, cortes, union_p, union_c])
    boton_dividir.click(dividir_app, [modelo, ultimo, tipo, plantilla, max_partes, cortes, union_p,
                                      union_c, diametro, holgura, altura, colores_d],
                        [informe, vista, archivos])

    with gr.Tab("Mecanismos"), gr.Row():
        with gr.Column():
            gr.Markdown("Convierte el modelo en un mecanismo que funciona: la IA (Claude) detecta "
                        "las piezas móviles (ruedas, brazos, bisagras), se les añaden ejes para "
                        "imprimirlas y se crea una animación. Necesita `ANTHROPIC_API_KEY`.")
            modelo_m = gr.File(label="Modelo (vacío = el último generado)",
                               file_types=[".glb", ".gltf", ".stl", ".obj", ".ply"], type="filepath")
            imagen_m = gr.Image(type="pil", label="Imagen original (opcional, ayuda a la IA)")
            descripcion_m = gr.Textbox(label="Pistas para la IA (opcional)",
                                       placeholder="p. ej. las 4 ruedas giran y el brazo lanza")
            altura_m = gr.Number(value=0, label="Altura (mm, 0 = la del modelo; un .glb se escala a 60)")
            boton_analizar = gr.Button("1. Analizar con IA", variant="primary")
            piezas_m = gr.Code(language="json", label="Piezas móviles (editable)")
            boton_vista_m = gr.Button("Actualizar vista previa")
            modo_m = gr.Radio(list(MODOS_MEC), value=list(MODOS_MEC)[0], label="Uniones")
            with gr.Row():
                holgura_m = gr.Number(value=0.3, label="Holgura (mm)")
                diametro_m = gr.Number(value=0, label="Diámetro del eje (mm, 0 = automático)")
            colores_m = gr.Slider(1, 8, value=1, step=1,
                                  label="Colores de filamento en el .3mf (1 = sin pintar; el .glb "
                                        "siempre lleva los colores del modelo)")
            boton_construir = gr.Button("2. Construir mecanismo", variant="primary")
        with gr.Column():
            informe_m = gr.Markdown()
            previa_m = gr.Image(label="Piezas detectadas (cajas sobre las vistas)")
            visor_m = gr.Model3D(label="Mecanismo animado")
            archivos_m = gr.File(label="Descargar (.glb animado, .zip con STL, .3mf)",
                                 file_count="multiple")
    boton_analizar.click(analizar_app, [modelo_m, ultimo, imagen_m, descripcion_m, altura_m],
                         [piezas_m, previa_m, informe_m])
    boton_vista_m.click(vista_app, [modelo_m, ultimo, piezas_m, altura_m], previa_m)
    boton_construir.click(construir_app, [modelo_m, ultimo, piezas_m, modo_m, holgura_m, diametro_m,
                                          altura_m, colores_m], [informe_m, visor_m, archivos_m])

if __name__ == "__main__":
    demo.queue().launch()
