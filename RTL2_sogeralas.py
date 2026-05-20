# -*- coding: utf-8 -*-

"""
=========================================================
RASTER -> LAS/LAZ ULTRA OTIMIZADO
=========================================================

RECURSOS:
- Streaming por blocos
- Suporta bilhões de pixels
- Baixo uso RAM
- Ignora NoData
- Escrita incremental
- Compressão LAZ
- Classe LAS baseada no valor do raster
- Divisão automática por classe

=========================================================
INSTALAR
=========================================================

pip install numpy rasterio laspy lazrs

=========================================================
CONFIGURE AQUI
=========================================================
"""

# =========================================================
# ENTRADA / SAÍDA
# =========================================================

INPUT_RASTER = r"D:\TESTES_PYTHON\RASTER_TO_LAS\classificado.tif"

OUTPUT_LAZ = r"D:\TESTES_PYTHON\RASTER_TO_LAS\result_full\saida.laz"

# =========================================================
# CONFIGURAÇÕES
# =========================================================

# Divide por valor do raster
# Ex:
# valor 1 -> saida_1.laz
# valor 2 -> saida_2.laz
DIVIDE = False

# Usa valor raster como Classification LAS
# Ideal para LiDAR360
CLASSID = True

# Classe fixa se CLASSID=False
FIXED_CLASS = 1

# Tamanho bloco
BLOCK_SIZE = 4096

# Escala LAS
SCALE_XY = 0.001
SCALE_Z = 0.001

# =========================================================

import os
import math
import time

import numpy as np
import rasterio
import laspy

from rasterio.windows import Window


# =========================================================


def create_header(src):

    header = laspy.LasHeader(
        point_format=6,
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
    except Exception:
        pass

    return header


# =========================================================


def process_block(src, window):

    arr = src.read(1, window=window)

    nodata = src.nodata

    if nodata is not None:
        mask = arr != nodata
    else:
        mask = np.isfinite(arr)

    if not np.any(mask):
        return None

    rows, cols = np.where(mask)

    values = arr[rows, cols]

    global_rows = rows + window.row_off
    global_cols = cols + window.col_off

    t = src.transform

    xs = t.c + (global_cols + 0.5) * t.a
    ys = t.f + (global_rows + 0.5) * t.e

    return (
        xs.astype(np.float64),
        ys.astype(np.float64),
        values.astype(np.int32)
    )


# =========================================================


def create_writer(path, header):

    return laspy.open(
        path,
        mode="w",
        header=header
    )


# =========================================================


def write_points(
        writer,
        header,
        xs,
        ys,
        zs,
        class_values
):

    count = len(xs)

    points = laspy.ScaleAwarePointRecord.zeros(
        count,
        header=header
    )

    points.x = xs
    points.y = ys
    points.z = zs

    points.classification = class_values.astype(np.uint8)

    writer.write_points(points)


# =========================================================


def main():

    start = time.time()

    print("=" * 60)
    print("ABRINDO RASTER")
    print("=" * 60)

    with rasterio.open(INPUT_RASTER) as src:

        width = src.width
        height = src.height

        total_pixels = width * height

        print(f"Largura : {width:,}")
        print(f"Altura  : {height:,}")
        print(f"Pixels  : {total_pixels:,}")

        header = create_header(src)

        n_cols = math.ceil(width / BLOCK_SIZE)
        n_rows = math.ceil(height / BLOCK_SIZE)

        total_blocks = n_cols * n_rows

        print(f"Blocos  : {total_blocks:,}")

        writers = {}

        # =================================================
        # MODO NORMAL
        # =================================================

        if not DIVIDE:

            writers["main"] = create_writer(
                OUTPUT_LAZ,
                header
            )

        total_written = 0
        processed_blocks = 0

        # =================================================

        for row_off in range(0, height, BLOCK_SIZE):

            block_h = min(BLOCK_SIZE, height - row_off)

            for col_off in range(0, width, BLOCK_SIZE):

                block_w = min(BLOCK_SIZE, width - col_off)

                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=block_w,
                    height=block_h
                )

                result = process_block(src, window)

                processed_blocks += 1

                if result is None:
                    continue

                xs, ys, vals = result

                # =========================================
                # CLASSIFICAÇÃO
                # =========================================

                if CLASSID:

                    class_values = vals.copy()

                    class_values[class_values < 0] = 0
                    class_values[class_values > 255] = 255

                else:

                    class_values = np.full(
                        len(vals),
                        FIXED_CLASS,
                        dtype=np.uint8
                    )

                # =========================================
                # DIVIDIR POR CLASSE
                # =========================================

                if DIVIDE:

                    unique_classes = np.unique(class_values)

                    for cls in unique_classes:

                        mask = class_values == cls

                        if not np.any(mask):
                            continue

                        if cls not in writers:

                            base = os.path.splitext(
                                OUTPUT_LAZ
                            )[0]

                            out_path = f"{base}_{cls}.laz"

                            print(
                                f"\nCriando arquivo: {out_path}"
                            )

                            writers[cls] = create_writer(
                                out_path,
                                header
                            )

                        write_points(
                            writers[cls],
                            header,
                            xs[mask],
                            ys[mask],
                            vals[mask],
                            class_values[mask]
                        )

                        total_written += np.count_nonzero(mask)

                # =========================================
                # ARQUIVO ÚNICO
                # =========================================

                else:

                    write_points(
                        writers["main"],
                        header,
                        xs,
                        ys,
                        vals,
                        class_values
                    )

                    total_written += len(xs)

                # =========================================

                elapsed = time.time() - start

                speed = (
                    total_written / elapsed
                    if elapsed > 0 else 0
                )

                percent = (
                    processed_blocks / total_blocks
                ) * 100

                print(
                    f"\r"
                    f"[{processed_blocks:,}/{total_blocks:,}] "
                    f"{percent:6.2f}% | "
                    f"Pontos: {total_written:,} | "
                    f"{speed:,.0f} pts/s",
                    end=""
                )

        # =================================================
        # FECHAR WRITERS
        # =================================================

        for w in writers.values():
            w.close()

    # =====================================================

    total_time = time.time() - start

    print("\n")
    print("=" * 60)
    print("FINALIZADO")
    print("=" * 60)

    print(f"Pontos gravados : {total_written:,}")
    print(f"Tempo total     : {total_time/60:.2f} min")

    if DIVIDE:
        print("Modo divisão    : ATIVO")
    else:
        print(f"Arquivo saída   : {OUTPUT_LAZ}")

    if CLASSID:
        print("Classification  : VALOR DO RASTER")
    else:
        print(f"Classification  : FIXO ({FIXED_CLASS})")


# =========================================================

if __name__ == "__main__":
    main()