"""Convierte un modelo en un mecanismo funcional: piezas móviles impresas y animación.

Uso:
    python mecanismo.py catapulta.glb --imagen catapulta.png
    python mecanismo.py catapulta.glb --piezas piezas.json --modo separadas

Sin --piezas, Claude analiza el modelo (y la imagen original si se da) y propone
las piezas móviles; las guarda en <modelo>_piezas.json para revisarlas y
reutilizarlas. Necesita una clave de la API de Anthropic (ANTHROPIC_API_KEY o
`ant auth login`) solo para ese análisis.

Salidas: <modelo>_mecanismo.glb (animado), .zip (STL por pieza y pasadores),
.3mf (montado o por piezas) y <modelo>_piezas.png (vista previa de las cajas).
"""
import argparse
import os

from PIL import Image

import mecanismos
from dividir import cargar


def main():
    p = argparse.ArgumentParser(description="Mecanismo funcional y animación a partir de un modelo")
    p.add_argument("modelo", help=".glb/.obj de la app o .stl de impresión")
    p.add_argument("--imagen", help="Imagen original (ayuda a la IA a entender el objeto)")
    p.add_argument("--descripcion", default="", help='Pistas para la IA, p. ej. "las 4 ruedas giran"')
    p.add_argument("--piezas", help="JSON con las piezas (en vez de pedírselas a la IA)")
    p.add_argument("--modo", default="montada", choices=["montada", "separadas"],
                   help="montada: imprime ya ensamblado (print-in-place); separadas: pasadores sueltos")
    p.add_argument("--holgura", type=float, default=0.3, help="Juego entre piezas móviles (mm)")
    p.add_argument("--diametro", type=float, default=0.0, help="Diámetro de los pasadores (0 = auto)")
    p.add_argument("--altura-mm", type=float, help="Altura final del modelo (mm)")
    p.add_argument("--colores", type=int, default=1, help="Filamentos para el 3MF pintado (.glb con textura)")
    args = p.parse_args()

    malla, textura = cargar(args.modelo, args.altura_mm)
    base = os.path.splitext(args.modelo)[0]
    if args.piezas:
        with open(args.piezas, encoding="utf-8") as f:
            piezas = mecanismos.desde_json(f.read())
    else:
        imagen = Image.open(args.imagen) if args.imagen else None
        piezas, resumen = mecanismos.analizar(malla, imagen, args.descripcion)
        print("IA:", resumen)
        with open(base + "_piezas.json", "w", encoding="utf-8") as f:
            f.write(mecanismos.a_json(piezas, resumen))
        print("Piezas guardadas en", base + "_piezas.json")
    for pz in piezas:
        print(f"- {pz.nombre}: {pz.tipo}, eje {pz.eje}, pivote {pz.pivote}. {pz.explicacion}")
    mecanismos.vista_previa(malla, piezas).save(base + "_piezas.png")
    mec = mecanismos.construir(malla, piezas, args.modo, args.holgura, args.diametro)
    mecanismos.colorear(mec, malla, textura)
    rutas = mecanismos.exportar(mec, base, args.modo, args.colores)
    for aviso in mec.avisos:
        print("Aviso:", aviso)
    print("Archivos:", ", ".join(rutas.values()), base + "_piezas.png")


if __name__ == "__main__":
    main()
