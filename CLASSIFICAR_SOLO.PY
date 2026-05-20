# -*- coding: utf-8 -*-

"""
==================================================================
RECLASSIFICADOR DE NUVEM DE PONTOS LAS/LAZ COM BASE EM RASTER
==================================================================

Lê um arquivo LAS/LAZ pronto e reclassifica seus pontos com base
nos valores de um raster de classificação (GeoTIFF).

ENTRADAS:
  - LAS/LAZ de entrada  (.las / .laz)
  - Raster de classif.  (.tif)  → valores inteiros (ex: 0,1,2,3...)

SAÍDA:
  - LAS/LAZ reclassificado com todos os atributos originais
    preservados (RGB, Z, intensidade, etc.)

INSTALAÇÃO:
  pip install numpy rasterio laspy lazrs pyproj

==================================================================
"""

# ================================================================
# CONFIGURAÇÃO – altere apenas esta seção
# ================================================================

INPUT_LAS  = r"D:\TESTES_PYTHON\RASTER_TO_LAS\NUVEM_METASHAPE.las"
CLASS_RASTER = r"D:\TESTES_PYTHON\RASTER_TO_LAS\CL2.tif"
OUTPUT_LAS = r"D:\TESTES_PYTHON\RASTER_TO_LAS\NUVEM_METASHAPE_reclassificada.las"

# Tamanho do lote de pontos processados por vez (reduza se faltar RAM)
# 5_000_000 = 5 milhões de pontos por vez (~200 MB RAM estimado)
BATCH_SIZE = 5_000_000

# Valor de classificação a usar quando o ponto cai fora do raster
# ou em pixel nodata. Padrão LAS: 1 = Unclassified
FALLBACK_CLASS = 1

# Se True, mantém a classificação original do LAS para pontos
# que caírem fora do raster (sobrescreve FALLBACK_CLASS nesses casos)
KEEP_ORIGINAL_IF_OUTSIDE = False

DEBUG = True

# ================================================================
# DICIONÁRIO DE CLASSIFICAÇÃO
# Mapeie os valores do raster para nomes legíveis (apenas para log).
# Formato:  valor_raster : (nome_legível, código_LAS_padrão)
# ================================================================

CLASS_DICT = {
    0 : ("Não classificado",  0),
    1 : ("Solo / Terreno",    2),
    2 : ("Vegetação baixa",   3),
    3 : ("Vegetação média",   4),
    4 : ("Vegetação alta",    5),
    5 : ("Edificação",        6),
    6 : ("Água",              9),
    # Adicione mais conforme necessário:
    # 7 : ("Estrada",         11),
    # 8 : ("Ruído",            7),
}

# ================================================================
# IMPORTS
# ================================================================

import gc
import math
import shutil
import time
import traceback

import numpy as np
import rasterio
from rasterio.transform import rowcol
import laspy

# ================================================================
# UTILITÁRIOS
# ================================================================

def log(msg: str):
    if DEBUG:
        print(msg)


def fmt(n):
    """Formata número com separador de milhar."""
    return f"{n:,}"


# ================================================================
# VALIDAÇÃO
# ================================================================

def validate(las_header, cls_src):
    print("\n" + "=" * 60)
    print("VALIDAÇÃO")
    print("=" * 60)

    print(f"\n[LAS/LAZ]")
    print(f"  Point Format  : {las_header.point_format.id}")
    print(f"  LAS Version   : {las_header.version}")
    print(f"  N° de pontos  : {fmt(las_header.point_count)}")
    print(f"  Scales        : {las_header.scales}")
    print(f"  Offsets       : {las_header.offsets}")

    # Tenta extrair CRS do LAS
    las_crs_name = "(desconhecido)"
    try:
        from pyproj import CRS as ProjCRS
        crs = ProjCRS.from_user_input(las_header.parse_crs())
        las_crs_name = f"{crs.name}  (EPSG:{crs.to_epsg()})"
    except Exception:
        pass
    print(f"  CRS           : {las_crs_name}")

    print(f"\n[RASTER CLASSIFICAÇÃO]")
    print(f"  Dimensões     : {cls_src.width:,} x {cls_src.height:,}")
    print(f"  Resolução     : {cls_src.res}")
    print(f"  Bounds        : {cls_src.bounds}")
    print(f"  Nodata        : {cls_src.nodata}")
    print(f"  CRS           : {cls_src.crs}")
    print(f"  Dtype         : {cls_src.dtypes[0]}")

    # Aviso se CRS parecem diferentes
    try:
        from pyproj import CRS as ProjCRS
        crs_las   = ProjCRS.from_user_input(las_header.parse_crs())
        crs_rast  = ProjCRS.from_user_input(cls_src.crs)
        if crs_las.to_epsg() != crs_rast.to_epsg():
            print("\n[⚠ AVISO] CRS do LAS e do raster parecem diferentes!")
            print(f"  LAS    : EPSG:{crs_las.to_epsg()}")
            print(f"  Raster : EPSG:{crs_rast.to_epsg()}")
            print("  Certifique-se de que as coordenadas estão no mesmo sistema.")
    except Exception:
        pass

    print("\nVALIDAÇÃO OK\n")


# ================================================================
# MAPEIA PONTOS XY → PIXEL DO RASTER  (vetorizado, sem loop)
# ================================================================

def classify_batch(xs: np.ndarray, ys: np.ndarray,
                   cls_src, orig_cls: np.ndarray) -> np.ndarray:
    """
    Dado um lote de coordenadas XY (arrays float64),
    retorna array uint8 com a classificação LAS para cada ponto.

    Pontos fora dos bounds ou em nodata recebem FALLBACK_CLASS
    (ou mantêm classificação original se KEEP_ORIGINAL_IF_OUTSIDE=True).
    """
    nodata_val = cls_src.nodata

    # Bounds do raster
    left   = cls_src.bounds.left
    right  = cls_src.bounds.right
    bottom = cls_src.bounds.bottom
    top    = cls_src.bounds.top

    # Máscara de pontos dentro dos bounds
    inside = (
        (xs >= left)  & (xs < right) &
        (ys > bottom) & (ys <= top)
    )

    # Converte XY → índice de pixel  (row, col) vetorizado
    # rasterio.transform.rowcol aceita arrays
    rows, cols = rowcol(cls_src.transform, xs[inside], ys[inside])

    # Garante que estão dentro da grade (clip defensivo)
    rows = np.clip(rows, 0, cls_src.height - 1)
    cols = np.clip(cols, 0, cls_src.width  - 1)

    # Lê os pixels de interesse em batch (window mínima)
    if len(rows) > 0:
        r_min, r_max = int(rows.min()), int(rows.max())
        c_min, c_max = int(cols.min()), int(cols.max())

        from rasterio.windows import Window
        win = Window(
            col_off=c_min,
            row_off=r_min,
            width=c_max - c_min + 1,
            height=r_max - r_min + 1,
        )
        tile = cls_src.read(1, window=win)

        # Ajusta índices relativos à janela
        local_rows = rows - r_min
        local_cols = cols - c_min
        raw_vals = tile[local_rows, local_cols].astype(np.float64)
    else:
        raw_vals = np.array([], dtype=np.float64)

    # Monta resultado final
    result = np.full(len(xs), FALLBACK_CLASS, dtype=np.int32)

    if KEEP_ORIGINAL_IF_OUTSIDE:
        # Mantém classe original para pontos fora
        result[~inside] = orig_cls[~inside].astype(np.int32)

    # Aplica valores do raster
    inside_idx = np.where(inside)[0]

    if nodata_val is not None:
        valid = (raw_vals != nodata_val) & np.isfinite(raw_vals)
    else:
        valid = np.isfinite(raw_vals)

    # Pontos com nodata → fallback (ou original)
    nd_mask = ~valid
    if KEEP_ORIGINAL_IF_OUTSIDE and np.any(nd_mask):
        result[inside_idx[nd_mask]] = orig_cls[inside_idx[nd_mask]].astype(np.int32)

    # Pontos válidos → valor do raster
    v_mask = valid
    if np.any(v_mask):
        result[inside_idx[v_mask]] = np.nan_to_num(
            raw_vals[v_mask], nan=FALLBACK_CLASS
        ).astype(np.int32)

    # Clip para uint8 (0–255, padrão LAS)
    return np.clip(result, 0, 255).astype(np.uint8)


# ================================================================
# RESUMO DE CLASSIFICAÇÃO
# ================================================================

def print_class_summary(title: str, cls_array: np.ndarray):
    print(f"\n[{title}]")
    print(f"  {'Valor':>6}  {'Rótulo':<22}  {'Cód.LAS':>7}  {'Pontos':>14}  {'%':>6}")
    print(f"  {'-'*6}  {'-'*22}  {'-'*7}  {'-'*14}  {'-'*6}")

    total = len(cls_array)
    unique_vals, counts = np.unique(cls_array, return_counts=True)

    for val, cnt in zip(unique_vals.tolist(), counts.tolist()):
        pct = cnt / total * 100 if total > 0 else 0.0
        label, las_code = CLASS_DICT.get(val, (f"Classe {val}", val))
        marker = "  ●" if cnt > 0 else "  ○"
        print(f"{marker} {val:>5}  {label:<22}  {las_code:>7}  {cnt:>14,}  {pct:>5.1f}%")

    print(f"\n  Total : {total:,} pontos")


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()

    print("\n" + "=" * 60)
    print("RECLASSIFICADOR LAS/LAZ  ←  RASTER")
    print("=" * 60)
    print(f"\n  Entrada  : {INPUT_LAS}")
    print(f"  Raster   : {CLASS_RASTER}")
    print(f"  Saída    : {OUTPUT_LAS}")
    print(f"  Batch    : {fmt(BATCH_SIZE)} pontos/lote")

    with rasterio.open(CLASS_RASTER) as cls_src:

        # ── Abre LAS para leitura (modo streaming) ─────────────
        with laspy.open(INPUT_LAS, mode="r") as las_reader:

            validate(las_reader.header, cls_src)

            total_pts    = las_reader.header.point_count
            n_batches    = math.ceil(total_pts / BATCH_SIZE)

            print(f"Total de pontos  : {fmt(total_pts)}")
            print(f"Lotes            : {n_batches}")
            print(f"\nIniciando reclassificação → {OUTPUT_LAS}\n")

            # Cria header de saída baseado no original
            out_header = laspy.LasHeader(
                point_format=las_reader.header.point_format,
                version=las_reader.header.version,
            )
            out_header.scales  = las_reader.header.scales
            out_header.offsets = las_reader.header.offsets

            # Copia VLRs (preserva CRS e outros metadados)
            try:
                out_header.vlrs = list(las_reader.header.vlrs)
            except Exception:
                pass

            # Acumuladores globais para log
            all_orig_cls = []
            all_new_cls  = []
            total_written = 0

            # ── Escrita streaming ──────────────────────────────
            with laspy.open(OUTPUT_LAS, mode="w", header=out_header) as las_writer:

                for batch_idx, pts in enumerate(
                    las_reader.chunk_iterator(BATCH_SIZE), start=1
                ):
                    t_batch = time.time()

                    xs = np.array(pts.x, dtype=np.float64)
                    ys = np.array(pts.y, dtype=np.float64)
                    orig_cls = np.array(pts.classification, dtype=np.uint8)
                    n = len(xs)

                    log(f"\nLote {batch_idx}/{n_batches}  "
                        f"({fmt(n)} pontos)  "
                        f"X:[{xs.min():.1f}…{xs.max():.1f}]  "
                        f"Y:[{ys.min():.1f}…{ys.max():.1f}]")

                    # ── Reclassifica ───────────────────────────
                    new_cls = classify_batch(xs, ys, cls_src, orig_cls)

                    # Estatísticas do lote
                    changed = int(np.sum(new_cls != orig_cls))
                    log(f"  Pontos alterados : {fmt(changed)} / {fmt(n)}  "
                        f"({changed/n*100:.1f}%)")

                    # Acumula para log global
                    all_orig_cls.append(orig_cls)
                    all_new_cls.append(new_cls)

                    # ── Copia buffer bruto e sobrescreve só classification ──
                    # pts.array é um numpy structured array — copiar é O(1) por campo.
                    raw = pts.array.copy()

                    # Localiza o nome exato do campo no dtype (case-insensitive)
                    cls_field = next(
                        (f for f in raw.dtype.names if f.lower() == "classification"),
                        None
                    )
                    if cls_field:
                        raw[cls_field] = new_cls
                    else:
                        # Fallback raro: sobrescreve via atributo
                        pts.classification = new_cls
                        raw = pts.array.copy()

                    new_pts = laspy.PackedPointRecord(
                        raw, point_format=out_header.point_format
                    )

                    # ── Verifica espaço em disco antes de gravar ───────
                    import os
                    out_drive = os.path.splitdrive(OUTPUT_LAS)[0] or "/"
                    free_gb = shutil.disk_usage(out_drive).free / 1e9
                    if free_gb < 1.0:
                        raise RuntimeError(
                            f"DISCO CHEIO! Apenas {free_gb:.2f} GB livres em {out_drive}. "
                            "Libere espaço e reinicie."
                        )

                    las_writer.write_points(new_pts)
                    total_written += n

                    elapsed = time.time() - t0
                    speed   = total_written / elapsed if elapsed > 0 else 1
                    pct     = total_written / total_pts * 100
                    eta_s   = (total_pts - total_written) / speed

                    log(f"  Progresso : {pct:.1f}%  |  "
                        f"Vel.: {speed:,.0f} pts/s  |  "
                        f"ETA: {eta_s/60:.1f} min  |  "
                        f"Lote: {time.time()-t_batch:.2f}s")

                    del pts, new_pts, xs, ys, orig_cls, new_cls
                    gc.collect()

    # ================================================================
    # RESUMO FINAL
    # ================================================================
    total_time = time.time() - t0

    all_orig = np.concatenate(all_orig_cls)
    all_new  = np.concatenate(all_new_cls)
    total_changed = int(np.sum(all_orig != all_new))

    print("\n" + "=" * 60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)
    print(f"  Arquivo gerado  : {OUTPUT_LAS}")
    print(f"  Pontos totais   : {fmt(total_written)}")
    print(f"  Pontos alterados: {fmt(total_changed)}  ({total_changed/total_written*100:.1f}%)")
    print(f"  Tempo total     : {total_time/60:.2f} min")
    print(f"  Velocidade média: {fmt(int(total_written/total_time))} pts/s")

    print_class_summary("CLASSIFICAÇÃO ORIGINAL (antes)", all_orig)
    print_class_summary("CLASSIFICAÇÃO NOVA     (depois)", all_new)

    print("\n" + "=" * 60)
    print("  Atributos preservados : TODOS (RGB, Z, Intensidade, etc.)")
    print("  Campo alterado        : classification")
    print(f"  Compatível com : LiDAR360  CloudCompare  PDAL  QGIS  ArcGIS Pro")
    print("=" * 60 + "\n")


# ================================================================

if __name__ == "__main__":
    main()