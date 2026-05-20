# -*- coding: utf-8 -*-
"""
Raster -> LAZ/LAS ultra otimizado (streaming por blocos)

Requer:
    pip install rasterio laspy numpy

Opcional para compressão LAZ:
    pip install lazrs

Uso:
    python raster_to_laz.py entrada.tif saida.laz

Características:
- Processa em blocos (não explode RAM)
- Suporta rasters gigantes (~ bilhões de pixels)
- Ignora NoData
- Escrita incremental
- Compressão LAZ automática
- Muito mais rápido que iterar pixel por pixel

IMPORTANTE:
- Cada pixel vira 1 ponto LAS/LAZ
- Coordenadas X/Y = centro do pixel
- Z = valor do raster
- Para 7 bilhões de pontos, espere arquivos MUITO grandes
- LAS tradicional possui limite prático. Use LAZ.
"""

import sys
import math
import numpy as np
import rasterio
import laspy

from rasterio.windows import Window


# =========================================================
# CONFIGURAÇÕES
# =========================================================

# Tamanho do bloco
# Ajuste conforme RAM disponível
BLOCK_SIZE = 2048

# Escalas LAS
# Menor = maior precisão = arquivo maior
SCALE_XY = 0.001
SCALE_Z = 0.001

# Classe LAS
CLASSIFICATION = 2  # Ground

# =========================================================


def create_header(bounds, crs_wkt):
    """
    Cria cabeçalho LAS otimizado para grandes volumes
    """

    header = laspy.LasHeader(
        point_format=6,
        version="1.4"
    )

    header.scales = np.array([SCALE_XY, SCALE_XY, SCALE_Z])

    header.offsets = np.array([
        bounds.left,
        bounds.bottom,
        0
    ])

    try:
        header.parse_crs(crs_wkt)
    except Exception:
        pass

    return header


def process_block(src, window):
    """
    Processa bloco do raster
    """

    arr = src.read(1, window=window, masked=True)

    if arr.mask.all():
        return None

    rows, cols = np.where(~arr.mask)

    if len(rows) == 0:
        return None

    values = arr.data[rows, cols]

    global_rows = rows + window.row_off
    global_cols = cols + window.col_off

    xs, ys = rasterio.transform.xy(
        src.transform,
        global_rows,
        global_cols,
        offset='center'
    )

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = values.astype(np.float64)

    return xs, ys, zs


def main(input_raster, output_las):

    print("=" * 60)
    print("ABRINDO RASTER")
    print("=" * 60)

    with rasterio.open(input_raster) as src:

        width = src.width
        height = src.height

        total_pixels = width * height

        print(f"Largura: {width}")
        print(f"Altura : {height}")
        print(f"Pixels : {total_pixels:,}")

        header = create_header(src.bounds, src.crs)

        # -------------------------------------------------
        # Escrita streaming
        # -------------------------------------------------

        with laspy.open(
            output_las,
            mode="w",
            header=header
        ) as writer:

            total_written = 0
            processed_blocks = 0

            n_cols = math.ceil(width / BLOCK_SIZE)
            n_rows = math.ceil(height / BLOCK_SIZE)

            total_blocks = n_cols * n_rows

            for row_off in range(0, height, BLOCK_SIZE):

                h = min(BLOCK_SIZE, height - row_off)

                for col_off in range(0, width, BLOCK_SIZE):

                    w = min(BLOCK_SIZE, width - col_off)

                    window = Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=w,
                        height=h
                    )

                    result = process_block(src, window)

                    processed_blocks += 1

                    if result is None:
                        continue

                    xs, ys, zs = result

                    count = len(xs)

                    las = laspy.ScaleAwarePointRecord.zeros(
                        count,
                        header=header
                    )

                    las.x = xs
                    las.y = ys
                    las.z = zs

                    # Supondo que você já tenha criado o objeto LAS com os pontos
                    num_points = len(las.x)  # ou las.header.point_count
                    las.classification = np.full(num_points, CLASSIFICATION, dtype=np.uint8)

                    writer.write_points(las)

                    total_written += count

                    print(
                        f"[{processed_blocks}/{total_blocks}] "
                        f"Pontos: {total_written:,}",
                        end="\r"
                    )

        print()
        print("=" * 60)
        print("FINALIZADO")
        print("=" * 60)
        print(f"Pontos gravados: {total_written:,}")
        print(f"Arquivo saída : {output_las}")


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Uso:")
        print("python raster_to_laz.py entrada.tif saida.laz")
        sys.exit(1)

    input_raster = sys.argv[1]
    output_las = sys.argv[2]

    main(input_raster, output_las)