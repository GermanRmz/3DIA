"""Prueba 1c: cortes planos automáticos para reducir cambios de filamento.

RESULTADO (ComfyUI_00025, 4 colores): no ahorra. Ningún corte simple ni
combinación de hasta 7 cortes horizontales baja de ~563 cambios, lo mismo que la
figura entera tumbada (Y arriba). Los 4 colores están mezclados a toda altura.

Idea: un cambio de filamento cuesta por capa y por color presente en esa capa.
Si la figura se corta en piezas y cada pieza se imprime por separado (y con el
eje que menos cambios le cause), muchas capas pasan a tener 1 o 2 colores.

Método (búsqueda codiciosa):
  1. Colores como la app: k-means a 4 + limpieza de manchas.
  2. Para cada pieza se prueban planos X/Y/Z cada 2 mm. El costo de una pieza es
     el mínimo de cambios estimados entre imprimirla con X, Y o Z hacia arriba.
     El relleno interior usa el filamento base de la pieza (el color presente en
     más capas), que se cuenta en todas sus capas.
  3. Se corta en el plano que más ahorra si el corte es pegable: sección >= 40 mm²
     y sin trozos sueltos de < 30 mm³. Se repite hasta que el ahorro sea poco.

No considera soportes ni la calidad de la superficie según la orientación: eso
lo decide luego el puntuador de orientación. Uso:
  python pruebas/corte_color.py modelo.glb [max_piezas]
"""
import os
import sys
import time
import zipfile

import manifold3d as m3
import numpy as np
import trimesh

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import acabados  # noqa: E402
import multicolor  # noqa: E402

SALIDA = os.path.join(RAIZ, "pruebas", "salida")
CAPA = 0.2
PASO = 2.0           # mm entre planos candidatos
MARGEN = 8.0         # mm mínimos de cada lado del corte
SECCION_MIN = 40.0   # mm² de superficie para pegar
TROZO_MIN = 30.0     # mm³; trozos sueltos menores invalidan el corte
AHORRO_MIN = 0.03    # fracción del total; por debajo se deja de cortar


# --- Estimación de cambios -------------------------------------------------------

def presencia(zmin, zmax, et, k):
    """Matriz capas x colores: True si el color toca esa capa."""
    if len(et) == 0:
        return np.zeros((0, k), bool)
    z0 = zmin.min()
    capas = int(np.ceil((zmax.max() - z0) / CAPA)) + 1
    a = np.clip(np.ceil((zmin - z0 - CAPA / 2) / CAPA), 0, capas).astype(np.int64)
    b = np.clip(np.floor((zmax - z0 - CAPA / 2) / CAPA) + 1, 0, capas).astype(np.int64)
    ok = b > a
    idx = np.r_[a[ok] * k + et[ok], b[ok] * k + et[ok]]
    peso = np.r_[np.ones(ok.sum()), -np.ones(ok.sum())]
    dif = np.bincount(idx, peso, (capas + 1) * k).reshape(capas + 1, k)
    return np.cumsum(dif, 0)[:capas] > 0.5


def cambios(pres):
    """Cambios con el filamento base presente en todas las capas de la pieza."""
    usadas = pres.any(1)
    if not usadas.any():
        return 0, 0
    base = int(pres.sum(0).argmax())
    p = pres[usadas].copy()
    p[:, base] = True
    n = p.sum(1)
    # Entre capas no hay cambio extra: siempre comparten el filamento base.
    return int((n - 1).sum()), base


def costo(tri, et, k):
    """(cambios, eje arriba, filamento base) con el mejor de los 3 ejes."""
    mejor = None
    for eje in range(3):
        c = tri[:, :, eje]
        n, base = cambios(presencia(c.min(1), c.max(1), et, k))
        if mejor is None or n < mejor[0]:
            mejor = (n, eje, base)
    return mejor


# --- Corte real con Manifold ---------------------------------------------------

def a_manifold(v, f):
    return m3.Manifold(m3.Mesh(vert_properties=v.astype(np.float32), tri_verts=f.astype(np.uint32),
                               face_id=np.arange(len(f), dtype=np.uint32)))


def de_manifold(man, et_origen, id_origen, relleno):
    """Malla trimesh + etiqueta por cara. Las tapas del corte llevan `relleno`."""
    me = man.to_mesh()
    v = np.asarray(me.vert_properties)[:, :3].astype(np.float64)
    f = np.asarray(me.tri_verts).astype(np.int64)
    fid = np.asarray(me.face_id).astype(np.int64)
    et = np.full(len(f), relleno, np.int64)
    ri = np.asarray(me.run_index) // 3
    for r, oid in enumerate(me.run_original_id):
        if oid == id_origen:
            et[ri[r]:ri[r + 1]] = et_origen[fid[ri[r]:ri[r + 1]]]
    return trimesh.Trimesh(v, f, process=False), et


class Pieza:
    def __init__(self, man, et_origen, id_origen, relleno, k):
        self.man = man
        self.malla, self.et = de_manifold(man, et_origen, id_origen, relleno)
        self.k = k
        self.tri = self.malla.triangles
        self.costo, self.eje, self.base = costo(self.tri, self.et, k)
        # Las tapas (caras de pegado) no se ven: toman el filamento base.
        self.et[self.et < 0] = self.base

    def candidatos(self):
        cen = self.tri.mean(1)
        lo, hi = self.malla.bounds
        for eje in range(3):
            for x in np.arange(lo[eje] + MARGEN, hi[eje] - MARGEN + 1e-6, PASO):
                m = cen[:, eje] < x
                ca = costo(self.tri[m], self.et[m], self.k)[0]
                cb = costo(self.tri[~m], self.et[~m], self.k)[0]
                yield self.costo - ca - cb, eje, x


def seccion(man, eje, x):
    """Área de la sección del sólido en el plano eje = x (mm²)."""
    if eje != 2:
        rot = [(0, -90, 0), (90, 0, 0)][eje]
        man = man.rotate(rot)  # rotación que deja el eje elegido como Z
    try:
        return man.slice(x).area()
    except Exception:
        return float(man.trim_by_plane((0, 0, -1), -x).trim_by_plane((0, 0, 1), x - 0.01).volume() / 0.01)


def intentar_corte(p, eje, x, id_origen):
    normal = [0.0, 0.0, 0.0]
    normal[eje] = 1.0
    if seccion(p.man, eje, x) < SECCION_MIN:
        return None, "sección chica"
    a, b = p.man.split_by_plane(normal, x)
    partes = []
    for lado in (b, a):  # b: lado negativo
        trozos = lado.decompose()
        vols = [t.volume() for t in trozos]
        if min(vols) < TROZO_MIN:
            return None, f"trozo suelto de {min(vols):.1f} mm³"
        partes.append(lado)
    return partes, None


# --- Salida 3MF con filamento base por objeto ------------------------------------

def exportar_pieza(ruta, malla, et, paleta, base):
    multicolor.exportar_3mf(ruta, malla, et, paleta)
    config = ('<?xml version="1.0" encoding="UTF-8"?>\n<config><object id="1">'
              f'<metadata key="extruder" value="{base + 1}"/></object></config>')
    with zipfile.ZipFile(ruta, "a") as z:
        z.writestr("Metadata/model_settings.config", config)


def orientar(malla, eje):
    """Pone `eje` hacia arriba y apoya la pieza en z = 0."""
    m = malla.copy()
    if eje == 0:
        m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
    elif eje == 1:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(-m.bounds[0])
    return m


def main():
    glb = sys.argv[1]
    max_piezas = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    textura = trimesh.load(glb, force="mesh")
    malla = acabados.aplicar(textura, "ninguno", 100)
    tx = textura.copy()
    tx.apply_transform(malla.metadata["transformacion"])
    centros = malla.metadata["vertices_base"][malla.faces].mean(axis=1)
    et, pal = multicolor.cuantizar(multicolor.colores_en_puntos(tx, centros), 4)
    et = multicolor.absorber_manchas(multicolor.suavizar(et, malla), malla)
    k = len(pal)

    man = a_manifold(malla.vertices, malla.faces)
    id_origen = man.to_mesh().run_original_id[0]
    piezas = [Pieza(man, et, id_origen, -1, k)]
    print(f"entera: cambios~{piezas[0].costo} (eje {'XYZ'[piezas[0].eje]} arriba)  {time.time() - t0:.0f} s")
    total0 = piezas[0].costo

    while len(piezas) < max_piezas:
        total = sum(p.costo for p in piezas)
        opciones = sorted(((g, i, e, x) for i, p in enumerate(piezas) for g, e, x in p.candidatos()),
                          reverse=True)
        hecho = False
        for g, i, e, x in opciones[:40]:
            if g < AHORRO_MIN * total:
                break
            partes, motivo = intentar_corte(piezas[i], e, x, id_origen)
            if partes is None:
                continue
            p = piezas.pop(i)
            nuevas = [Pieza(m, et, id_origen, -1, k) for m in partes]
            piezas += nuevas
            print(f"corte {'XYZ'[e]}={x:.0f} mm en pieza {i}: ahorro estimado {g}; "
                  f"ahora {sum(q.costo for q in piezas)}  ({time.time() - t0:.0f} s)")
            hecho = True
            break
        if not hecho:
            break

    base = os.path.join(SALIDA, os.path.splitext(os.path.basename(glb))[0] + "_corte")
    print(f"\n{'pieza':>5} {'medidas mm':>18} {'colores':>8} {'base':>4} {'arriba':>6} {'cambios~':>8}")
    for n, p in enumerate(piezas, 1):
        colores = np.unique(p.et).size
        m = orientar(p.malla, p.eje)
        exportar_pieza(f"{base}_{n}.3mf", m, p.et, pal, p.base)
        print(f"{n:>5} {str(p.malla.extents.round(0).tolist()):>18} {colores:>8} {p.base + 1:>4} "
              f"{'XYZ'[p.eje]:>6} {p.costo:>8}")
    print(f"total cambios~ {sum(p.costo for p in piezas)} (entera {total0})  {time.time() - t0:.0f} s")

    # Vista con las piezas separadas (explosionada) para revisar los cortes.
    escena, c0 = [], malla.bounds.mean(0)
    for p in piezas:
        v = trimesh.Trimesh(p.malla.vertices, p.malla.faces, process=False)
        v.visual.face_colors = np.column_stack([pal[p.et], np.full(len(p.et), 255)])
        v.apply_translation((p.malla.bounds.mean(0) - c0) * 0.35)
        escena.append(v)
    vista = trimesh.util.concatenate(escena)
    vista.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    vista.export(base + "_vista.glb")


if __name__ == "__main__":
    main()
