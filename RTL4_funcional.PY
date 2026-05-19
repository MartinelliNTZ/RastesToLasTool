# -*- coding: utf-8 -*-

"""
==================================================================
SUPER RASTER -> LAZ/LAS COLORIDO + ALTURA + CLASSIFICAÇÃO
==================================================================

VERSÃO OTIMIZADA - SEM dsm.sample() (gargalo eliminado)

ENTRADAS:
  - ORTOMOSAICO RGB  (.tif)
  - DSM/MDS          (.tif)
  - CLASSIFICAÇÃO    (.tif)  → valores 0,1,2,3... ou 1,2,3...

SAÍDA:
  - LAZ colorido (RGB real) com Z = cota DSM e Classification LAS

INSTALAÇÃO:
  pip install numpy rasterio laspy lazrs

==================================================================
"""

# ================================================================
# CONFIGURAÇÃO – altere apenas esta seção
# ================================================================

ORTHO  = r"D:\TESTES_PYTHON\RASTER_TO_LAS\imaru2.tif"
DSM    = r"D:\TESTES_PYTHON\RASTER_TO_LAS\ImanuMDS.tif"
CLASS  = r"D:\TESTES_PYTHON\RASTER_TO_LAS\classificado2.tif"
OUTPUT = r"D:\TESTES_PYTHON\RASTER_TO_LAS\nuvem_final2.las"

BLOCK_SIZE = 1024        # pixels por bloco (reduza se faltar RAM)

SCALE_XY   = 0.001       # precisão XY em metros
SCALE_Z    = 0.001       # precisão Z  em metros

SKIP_BLACK = False        # True → ignora pixels totalmente pretos

DEBUG      = True

# ================================================================
# IMPORTS
# ================================================================

import gc
import math
import time
import traceback

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import laspy


# ================================================================
# UTILITÁRIOS
# ================================================================

def log(msg: str):
    if DEBUG:
        print(msg)


def lerp_window_to_other(src_ortho, src_other, window: Window) -> Window:
    """
    Converte uma Window (em pixels do ortomosaico) para a janela
    equivalente em pixels de OUTRO raster (DSM ou CLASS), que pode ter
    resolução e extensão diferentes.
    """
    # Bounding box geográfico do bloco no ortomosaico
    left, bottom, right, top = rasterio.windows.bounds(
        window, src_ortho.transform
    )

    # Converte para Window no outro raster
    win = rasterio.windows.from_bounds(
        left, bottom, right, top,
        transform=src_other.transform
    )

    # Garante que fica dentro do arquivo
    win = win.intersection(
        Window(0, 0, src_other.width, src_other.height)
    )

    return win


# ================================================================
# LEITURA DE BLOCO DSM/CLASS REPROJETADO PARA GRADE DO ORTHO
# ================================================================

def read_resampled(src_other, window_other: Window, out_shape) -> np.ndarray:
    """
    Lê src_other na window_other e reamostraz para out_shape (H, W).
    Retorna array 2D float32/uint8.
    """
    h, w = out_shape

    if window_other.width <= 0 or window_other.height <= 0:
        return np.full((h, w), np.nan, dtype=np.float32)

    data = src_other.read(
        1,
        window=window_other,
        out_shape=(h, w),
        resampling=Resampling.nearest,
        boundless=True,
        fill_value=np.nan if src_other.nodata is None else src_other.nodata,
    )
    return data


# ================================================================
# HEADER LAS
# ================================================================

def create_header(src_ortho):
    header = laspy.LasHeader(point_format=7, version="1.4")

    header.scales  = np.array([SCALE_XY, SCALE_XY, SCALE_Z])
    header.offsets = np.array([
        src_ortho.bounds.left,
        src_ortho.bounds.bottom,
        0.0,
    ])

    try:
        header.add_crs(src_ortho.crs)
        log(f"CRS gravado no header: {src_ortho.crs}")
    except Exception as e:
        print(f"[AVISO] Falha ao gravar CRS: {e}")

    return header


# ================================================================
# VALIDAÇÃO
# ================================================================

def validate(ortho, dsm, cls):
    print("\n" + "=" * 60)
    print("VALIDAÇÃO DOS RASTERS")
    print("=" * 60)

    print(f"\n[CRS]")
    print(f"  ORTHO : {ortho.crs}")
    print(f"  DSM   : {dsm.crs}")
    print(f"  CLASS : {cls.crs}")

    if ortho.crs and dsm.crs and (ortho.crs != dsm.crs):
        raise ValueError("ORTHO e DSM têm CRS diferentes!")
    if ortho.crs and cls.crs and (ortho.crs != cls.crs):
        raise ValueError("ORTHO e CLASS têm CRS diferentes!")

    print(f"\n[RESOLUÇÃO]")
    print(f"  ORTHO : {ortho.res}")
    print(f"  DSM   : {dsm.res}")
    print(f"  CLASS : {cls.res}")

    print(f"\n[DIMENSÕES]")
    print(f"  ORTHO : {ortho.width:,} x {ortho.height:,}")
    print(f"  DSM   : {dsm.width:,} x {dsm.height:,}")
    print(f"  CLASS : {cls.width:,} x {cls.height:,}")

    print(f"\n[BOUNDS]")
    print(f"  ORTHO : {ortho.bounds}")
    print(f"  DSM   : {dsm.bounds}")
    print(f"  CLASS : {cls.bounds}")

    # Nodata dos rasters auxiliares
    print(f"\n[NODATA]")
    print(f"  DSM   : {dsm.nodata}")
    print(f"  CLASS : {cls.nodata}")

    print("\nVALIDAÇÃO OK\n")


# ================================================================
# PROCESSAR BLOCO  (sem .sample – tudo por índice de pixel)
# ================================================================

def process_block(ortho, dsm, cls, window: Window):
    bh = window.height
    bw = window.width
    out_shape = (bh, bw)

    # ----------------------------------------------------------
    # 1. RGB do ortomosaico
    # ----------------------------------------------------------
    r = ortho.read(1, window=window).astype(np.float32)
    g = ortho.read(2, window=window).astype(np.float32)
    b = ortho.read(3, window=window).astype(np.float32)

    # ----------------------------------------------------------
    # 2. DSM reamostrado para a grade do ortomosaico
    # ----------------------------------------------------------
    win_dsm   = lerp_window_to_other(ortho, dsm,   window)
    win_class = lerp_window_to_other(ortho, cls,   window)

    z_grid = read_resampled(dsm, win_dsm,   out_shape).astype(np.float32)
    c_grid = read_resampled(cls, win_class, out_shape).astype(np.float32)

    # ----------------------------------------------------------
    # 3. Máscara de pixels válidos
    # ----------------------------------------------------------
    nodata_dsm = dsm.nodata if dsm.nodata is not None else -9999.0

    mask = (
        np.isfinite(r) &
        np.isfinite(g) &
        np.isfinite(b) &
        np.isfinite(z_grid) &
        (z_grid != nodata_dsm)
    )

    if SKIP_BLACK:
        mask &= (r > 0) | (g > 0) | (b > 0)

    if not np.any(mask):
        return None

    # ----------------------------------------------------------
    # 4. Coordenadas geográficas (vetorizadas)
    # ----------------------------------------------------------
    rows, cols = np.where(mask)

    global_rows = rows + window.row_off
    global_cols = cols + window.col_off

    t  = ortho.transform
    xs = t.c + (global_cols + 0.5) * t.a
    ys = t.f + (global_rows + 0.5) * t.e

    # ----------------------------------------------------------
    # 5. Extrai valores por índice – SEM .sample()
    # ----------------------------------------------------------
    zs  = z_grid[rows, cols].astype(np.float64)
    cls_vals = c_grid[rows, cols]

    # Limpa nodata residual no Z
    valid_z = np.isfinite(zs) & (zs != nodata_dsm)
    if not np.any(valid_z):
        return None

    xs       = xs[valid_z]
    ys       = ys[valid_z]
    zs       = zs[valid_z]
    cls_vals = cls_vals[valid_z]

    rr = r[rows, cols][valid_z]
    gg = g[rows, cols][valid_z]
    bb = b[rows, cols][valid_z]

    # ----------------------------------------------------------
    # 6. Classificação LAS (0–255)
    # ----------------------------------------------------------
    cls_las = np.clip(
        np.nan_to_num(cls_vals, nan=1).astype(np.int32),
        0, 255
    ).astype(np.uint8)

    return (
        xs.astype(np.float64),
        ys.astype(np.float64),
        zs.astype(np.float64),
        rr.astype(np.uint16),
        gg.astype(np.uint16),
        bb.astype(np.uint16),
        cls_las,
    )


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()

    print("\n" + "=" * 60)
    print("RASTER → LAZ  (versão otimizada)")
    print("=" * 60)

    with rasterio.open(ORTHO) as ortho, \
         rasterio.open(DSM)   as dsm,   \
         rasterio.open(CLASS) as cls:

        validate(ortho, dsm, cls)

        W, H = ortho.width, ortho.height
        total_px = W * H

        header = create_header(ortho)

        n_cols = math.ceil(W / BLOCK_SIZE)
        n_rows = math.ceil(H / BLOCK_SIZE)
        total_blocks = n_cols * n_rows

        print(f"Dimensões ortho   : {W:,} x {H:,}  ({total_px:,} pixels)")
        print(f"Blocos X          : {n_cols:,}")
        print(f"Blocos Y          : {n_rows:,}")
        print(f"Total blocos      : {total_blocks:,}")
        print(f"\nIniciando escrita → {OUTPUT}\n")

        total_written   = 0
        processed       = 0
        min_z = max_z   = None
        global_classes  = set()

        with laspy.open(OUTPUT, mode="w", header=header) as writer:

            for row_off in range(0, H, BLOCK_SIZE):
                bh = min(BLOCK_SIZE, H - row_off)

                for col_off in range(0, W, BLOCK_SIZE):
                    bw = min(BLOCK_SIZE, W - col_off)
                    processed += 1
                    t_block = time.time()

                    window = Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=bw,
                        height=bh,
                    )

                    log(f"\n{'='*60}")
                    log(f"BLOCO {processed:,}/{total_blocks:,}  "
                        f"ROW={row_off:,} COL={col_off:,}  "
                        f"SIZE={bw:,}x{bh:,}")

                    try:
                        result = process_block(ortho, dsm, cls, window)
                    except Exception:
                        print(f"[ERRO] Bloco {processed} falhou:")
                        traceback.print_exc()
                        continue

                    if result is None:
                        log("  → bloco vazio, pulado")
                        continue

                    xs, ys, zs, rs, gs, bs, cls_vals = result
                    n = len(xs)

                    # ----------------------------------------
                    # Escrita LAS
                    # ----------------------------------------
                    pts = laspy.ScaleAwarePointRecord.zeros(n, header=header)
                    pts.x              = xs
                    pts.y              = ys
                    pts.z              = zs
                    pts.red            = rs * 256   # escala 8→16 bit
                    pts.green          = gs * 256
                    pts.blue           = bs * 256
                    pts.classification = cls_vals

                    writer.write_points(pts)

                    # ----------------------------------------
                    # Estatísticas
                    # ----------------------------------------
                    total_written += n
                    global_classes.update(np.unique(cls_vals).tolist())

                    lmin, lmax = float(zs.min()), float(zs.max())
                    min_z = lmin if min_z is None else min(min_z, lmin)
                    max_z = lmax if max_z is None else max(max_z, lmax)

                    elapsed = time.time() - t0
                    speed   = total_written / elapsed if elapsed > 0 else 0
                    pct     = processed / total_blocks * 100
                    rem     = total_blocks - processed
                    eta_s   = (rem * (total_written / processed)) / speed if speed > 0 else 0

                    log(f"  Pontos bloco : {n:,}")
                    log(f"  Pontos total : {total_written:,}")
                    log(f"  Z bloco      : [{lmin:.3f} … {lmax:.3f}]")
                    log(f"  Classes      : {np.unique(cls_vals).tolist()}")
                    log(f"  Velocidade   : {speed:,.0f} pts/s")
                    log(f"  Tempo bloco  : {time.time()-t_block:.2f} s")
                    log(f"  Progresso    : {pct:.1f}%  ETA {eta_s/60:.1f} min")

                    del pts
                    gc.collect()

    # ================================================================
    # RESUMO FINAL
    # ================================================================
    total_time = time.time() - t0

    print("\n" + "=" * 60)
    print("FINALIZADO")
    print("=" * 60)
    print(f"  Arquivo        : {OUTPUT}")
    print(f"  Pontos totais  : {total_written:,}")
    print(f"  Tempo total    : {total_time/60:.2f} min")
    print(f"  Vel. média     : {total_written/total_time:,.0f} pts/s")
    print(f"  Z MIN          : {min_z:.3f}")
    print(f"  Z MAX          : {max_z:.3f}")
    print(f"  Classes LAS    : {sorted(global_classes)}")
    print()
    print("  Atributos      : X  Y  Z  Red  Green  Blue  Classification")
    print("  Point Format   : 7  |  LAS 1.4")
    print("  Compatível com : LiDAR360  CloudCompare  PDAL  QGIS  ArcGIS Pro")
    print("=" * 60)


# ================================================================

if __name__ == "__main__":
    main()