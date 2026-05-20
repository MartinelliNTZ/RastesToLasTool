# -*- coding: utf-8 -*-

"""
==================================================================
SUPER RASTER -> LAZ/LAS COLORIDO + ALTURA + CLASSIFICAÇÃO
==================================================================

PIPELINE PROFISSIONAL PARA GERAÇÃO DE NUVEM LAS/LAZ

ENTRADAS:
- ORTOMOSAICO RGB
- DSM/MDS
- CLASSIFICAÇÃO

SAÍDA:
- LAS/LAZ COLORIDO
- Z REAL
- CLASSIFICATION LAS
- COMPATÍVEL LiDAR360

==================================================================
RECURSOS
==================================================================

✔ Streaming por blocos
✔ Ultra otimizado
✔ Feedback completo
✔ Estatísticas detalhadas
✔ Compatível bilhões de pixels
✔ RGB real
✔ Classification LAS
✔ DSM amostrado espacialmente
✔ CRS automático
✔ Debug completo
✔ ETA
✔ Velocidade
✔ Estatísticas por bloco
✔ Resoluções diferentes
✔ Dimensões diferentes
✔ Amostragem espacial profissional

==================================================================
INSTALAR
==================================================================

pip install numpy rasterio laspy lazrs

==================================================================
"""

# ================================================================
# CONFIGURAÇÃO
# ================================================================

ORTHO = r"D:\TESTES_PYTHON\RASTER_TO_LAS\imaru2.tif"

DSM = r"D:\TESTES_PYTHON\RASTER_TO_LAS\ImanuMDS.tif"

CLASS = r"D:\TESTES_PYTHON\RASTER_TO_LAS\classificado2.tif"

OUTPUT = r"D:\TESTES_PYTHON\RASTER_TO_LAS\nuvem_final.laz"

# ================================================================
# CONFIG
# ================================================================

BLOCK_SIZE = 2048

SCALE_XY = 0.001
SCALE_Z = 0.001

USE_CLASSIFICATION = True

FIXED_CLASS = 1

SKIP_BLACK = False

DEBUG = True

# ================================================================

import os
import gc
import math
import time

import numpy as np
import rasterio
import laspy

from rasterio.windows import Window


# ================================================================
# DEBUG
# ================================================================

def debug(msg):

    if DEBUG:
        print(msg)


# ================================================================
# HEADER
# ================================================================

def create_header(src):

    header = laspy.LasHeader(
        point_format=7,
        version="1.4"
    )

    header.scales = np.array([
        SCALE_XY,
        SCALE_XY,
        SCALE_Z
    ])

    header.offsets = np.array([
        src.bounds.left,
        src.bounds.bottom,
        0
    ])

    try:
        header.add_crs(src.crs)

    except Exception as e:

        print(f"Falha CRS: {e}")

    return header


# ================================================================
# VALIDAR
# ================================================================

def validate_rasters(
        ortho,
        dsm,
        cls
):

    print("\n")
    print("=" * 70)
    print("VALIDAÇÃO DOS RASTERS")
    print("=" * 70)

    # ============================================================
    # CRS
    # ============================================================

    print("\n[CRS]")

    print(f"ORTHO : {ortho.crs}")
    print(f"DSM   : {dsm.crs}")
    print(f"CLASS : {cls.crs}")

    if ortho.crs != dsm.crs:
        raise Exception(
            "ORTHO e DSM possuem CRS diferentes"
        )

    if ortho.crs != cls.crs:
        raise Exception(
            "ORTHO e CLASS possuem CRS diferentes"
        )

    print("CRS OK")

    # ============================================================
    # RESOLUÇÃO
    # ============================================================

    print("\n[RESOLUÇÃO]")

    print(f"ORTHO : {ortho.res}")
    print(f"DSM   : {dsm.res}")
    print(f"CLASS : {cls.res}")

    # ============================================================
    # DIMENSÕES
    # ============================================================

    print("\n[DIMENSÕES]")

    print(
        f"ORTHO : {ortho.width:,} x {ortho.height:,}"
    )

    print(
        f"DSM   : {dsm.width:,} x {dsm.height:,}"
    )

    print(
        f"CLASS : {cls.width:,} x {cls.height:,}"
    )

    # ============================================================
    # EXTENSÃO
    # ============================================================

    print("\n[BOUNDS]")

    print(f"ORTHO : {ortho.bounds}")
    print(f"DSM   : {dsm.bounds}")
    print(f"CLASS : {cls.bounds}")

    print("\nVALIDAÇÃO FINALIZADA")


# ================================================================
# PROCESSAR BLOCO
# ================================================================

def process_block(
        ortho,
        dsm,
        cls,
        window
):

    # ============================================================
    # RGB
    # ============================================================

    r = ortho.read(
        1,
        window=window
    )

    g = ortho.read(
        2,
        window=window
    )

    b = ortho.read(
        3,
        window=window
    )

    # ============================================================
    # MÁSCARA
    # ============================================================

    mask = (
        np.isfinite(r) &
        np.isfinite(g) &
        np.isfinite(b)
    )

    if SKIP_BLACK:

        rgb_mask = (
            (r > 0) |
            (g > 0) |
            (b > 0)
        )

        mask &= rgb_mask

    if not np.any(mask):
        return None

    rows, cols = np.where(mask)

    rr = r[rows, cols]
    gg = g[rows, cols]
    bb = b[rows, cols]

    # ============================================================
    # COORDENADAS
    # ============================================================

    global_rows = rows + window.row_off
    global_cols = cols + window.col_off

    t = ortho.transform

    xs = (
        t.c +
        (global_cols + 0.5) * t.a
    )

    ys = (
        t.f +
        (global_rows + 0.5) * t.e
    )

    coords = np.column_stack([
        xs,
        ys
    ])

    # ============================================================
    # SAMPLE DSM
    # ============================================================

    dsm_values = np.array([
        v[0]
        for v in dsm.sample(coords)
    ])

    # ============================================================
    # SAMPLE CLASS
    # ============================================================

    class_values = np.array([
        v[0]
        for v in cls.sample(coords)
    ])

    # ============================================================
    # FILTRAR DSM
    # ============================================================

    valid = np.isfinite(dsm_values)

    if not np.any(valid):
        return None

    xs = xs[valid]
    ys = ys[valid]

    rr = rr[valid]
    gg = gg[valid]
    bb = bb[valid]

    dsm_values = dsm_values[valid]

    class_values = class_values[valid]

    return (

        xs.astype(np.float64),
        ys.astype(np.float64),

        dsm_values.astype(np.float64),

        rr.astype(np.uint16),
        gg.astype(np.uint16),
        bb.astype(np.uint16),

        class_values.astype(np.uint8)
    )


# ================================================================
# MAIN
# ================================================================

def main():

    start = time.time()

    print("\n")
    print("=" * 70)
    print("SUPER RASTER -> LAZ")
    print("=" * 70)

    # ============================================================
    # OPEN
    # ============================================================

    print("\nAbrindo rasters...")

    with rasterio.open(ORTHO) as ortho, \
         rasterio.open(DSM) as dsm, \
         rasterio.open(CLASS) as cls:

        # ========================================================
        # VALIDAR
        # ========================================================

        validate_rasters(
            ortho,
            dsm,
            cls
        )

        # ========================================================
        # INFO
        # ========================================================

        width = ortho.width
        height = ortho.height

        total_pixels = width * height

        print("\n")
        print("=" * 70)
        print("ESTATÍSTICAS")
        print("=" * 70)

        print(f"Pixels totais : {total_pixels:,}")

        # ========================================================
        # HEADER
        # ========================================================

        print("\nCriando header LAS...")

        header = create_header(ortho)

        # ========================================================
        # BLOCOS
        # ========================================================

        n_cols = math.ceil(
            width / BLOCK_SIZE
        )

        n_rows = math.ceil(
            height / BLOCK_SIZE
        )

        total_blocks = n_cols * n_rows

        print(f"Blocos X      : {n_cols:,}")
        print(f"Blocos Y      : {n_rows:,}")
        print(f"Total blocos  : {total_blocks:,}")

        # ========================================================
        # ESCRITA
        # ========================================================

        print("\nIniciando escrita LAZ...\n")

        total_written = 0

        processed_blocks = 0

        min_z = None
        max_z = None

        global_classes = set()

        with laspy.open(
                OUTPUT,
                mode="w",
                header=header
        ) as writer:

            # ====================================================
            # LOOP LINHAS
            # ====================================================

            for row_off in range(
                    0,
                    height,
                    BLOCK_SIZE
            ):

                block_h = min(
                    BLOCK_SIZE,
                    height - row_off
                )

                # =================================================
                # LOOP COLUNAS
                # =================================================

                for col_off in range(
                        0,
                        width,
                        BLOCK_SIZE
                ):

                    block_w = min(
                        BLOCK_SIZE,
                        width - col_off
                    )

                    processed_blocks += 1

                    block_start = time.time()

                    # =============================================
                    # WINDOW
                    # =============================================

                    window = Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=block_w,
                        height=block_h
                    )

                    debug("\n")
                    debug("=" * 70)

                    debug(
                        f"BLOCO "
                        f"{processed_blocks:,}/{total_blocks:,}"
                    )

                    debug(
                        f"ROW={row_off:,} "
                        f"COL={col_off:,}"
                    )

                    debug(
                        f"SIZE={block_w:,}x{block_h:,}"
                    )

                    # =============================================
                    # PROCESSAR
                    # =============================================

                    result = process_block(
                        ortho,
                        dsm,
                        cls,
                        window
                    )

                    if result is None:

                        debug(
                            "Bloco vazio"
                        )

                        continue

                    (
                        xs,
                        ys,
                        zs,
                        rs,
                        gs,
                        bs,
                        cls_vals
                    ) = result

                    count = len(xs)

                    # =============================================
                    # CLASS
                    # =============================================

                    if USE_CLASSIFICATION:

                        class_values = cls_vals.copy()

                        class_values[
                            class_values < 0
                        ] = 0

                        class_values[
                            class_values > 255
                        ] = 255

                    else:

                        class_values = np.full(
                            count,
                            FIXED_CLASS,
                            dtype=np.uint8
                        )

                    # =============================================
                    # LAS
                    # =============================================

                    points = laspy.ScaleAwarePointRecord.zeros(
                        count,
                        header=header
                    )

                    points.x = xs
                    points.y = ys
                    points.z = zs

                    points.red = rs * 256
                    points.green = gs * 256
                    points.blue = bs * 256

                    points.classification = (
                        class_values
                    )

                    # =============================================
                    # WRITE
                    # =============================================

                    writer.write_points(points)

                    # =============================================
                    # STATS
                    # =============================================

                    total_written += count

                    global_classes.update(
                        np.unique(class_values)
                    )

                    local_min_z = float(
                        np.min(zs)
                    )

                    local_max_z = float(
                        np.max(zs)
                    )

                    if min_z is None:

                        min_z = local_min_z
                        max_z = local_max_z

                    else:

                        min_z = min(
                            min_z,
                            local_min_z
                        )

                        max_z = max(
                            max_z,
                            local_max_z
                        )

                    elapsed = (
                        time.time() - start
                    )

                    speed = (
                        total_written / elapsed
                    )

                    percent = (
                        processed_blocks /
                        total_blocks
                    ) * 100

                    remaining = (
                        total_blocks -
                        processed_blocks
                    )

                    avg_pts_block = (
                        total_written /
                        processed_blocks
                    )

                    eta_seconds = (
                        remaining *
                        avg_pts_block
                    ) / speed

                    eta_min = (
                        eta_seconds / 60
                    )

                    block_time = (
                        time.time() -
                        block_start
                    )

                    # =============================================
                    # DEBUG
                    # =============================================

                    debug(
                        f"Pontos bloco : {count:,}"
                    )

                    debug(
                        f"Pontos total : "
                        f"{total_written:,}"
                    )

                    debug(
                        f"Velocidade   : "
                        f"{speed:,.0f} pts/s"
                    )

                    debug(
                        f"Tempo bloco  : "
                        f"{block_time:.2f} s"
                    )

                    debug(
                        f"ETA          : "
                        f"{eta_min:.2f} min"
                    )

                    debug(
                        f"Min Z bloco  : "
                        f"{local_min_z:.3f}"
                    )

                    debug(
                        f"Max Z bloco  : "
                        f"{local_max_z:.3f}"
                    )

                    debug(
                        f"Classes      : "
                        f"{np.unique(class_values)}"
                    )

                    debug(
                        f"Progresso    : "
                        f"{percent:.2f}%"
                    )

                    debug("=" * 70)

                    # =============================================
                    # LIMPEZA
                    # =============================================

                    del points

                    gc.collect()

    # ============================================================
    # FINAL
    # ============================================================

    total_time = (
        time.time() - start
    )

    print("\n")
    print("=" * 70)
    print("FINALIZADO")
    print("=" * 70)

    print(f"\nArquivo        : {OUTPUT}")

    print(
        f"Pontos         : "
        f"{total_written:,}"
    )

    print(
        f"Tempo total    : "
        f"{total_time/60:.2f} min"
    )

    print(
        f"Velocidade média: "
        f"{total_written/total_time:,.0f} pts/s"
    )

    print(f"\nZ MIN          : {min_z:.3f}")
    print(f"Z MAX          : {max_z:.3f}")

    print(
        f"\nClasses LAS    : "
        f"{sorted(global_classes)}"
    )

    print("\n")
    print("=" * 70)
    print("ATRIBUTOS LAS")
    print("=" * 70)

    print("X")
    print("Y")
    print("Z")
    print("Red")
    print("Green")
    print("Blue")
    print("Classification")

    print("\n")
    print("=" * 70)
    print("METADADOS")
    print("=" * 70)

    print(f"Point Format : 7")
    print(f"LAS Version  : 1.4")

    print(f"Scale XY     : {SCALE_XY}")
    print(f"Scale Z      : {SCALE_Z}")

    print(f"CRS          : {ortho.crs}")

    print("\n")
    print("=" * 70)
    print("COMPATIBILIDADE")
    print("=" * 70)

    print("LiDAR360")
    print("CloudCompare")
    print("PDAL")
    print("QGIS")
    print("ArcGIS Pro")

    print("\n")
    print("=" * 70)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)


# ================================================================
# START
# ================================================================

if __name__ == "__main__":

    main()