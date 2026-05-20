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

ORTHO  = r"D:\TESTES_PYTHON\RASTER_TO_LAS\Ortomosaico2.tif"
DSM    = r"D:\TESTES_PYTHON\RASTER_TO_LAS\ImanuMDS.tif"
CLASS  = r"D:\TESTES_PYTHON\RASTER_TO_LAS\classificado2.tif"
OUTPUT = r"D:\TESTES_PYTHON\RASTER_TO_LAS\IMARU_teste_rtl6.las"

BLOCK_SIZE = 1024*8        # pixels por bloco (reduza se faltar RAM)

SCALE_XY   = 0.001       # precisão XY em metros
SCALE_Z    = 0.001       # precisão Z  em metros

SKIP_BLACK = False        # True → ignora pixels totalmente pretos

DEBUG      = True

# ================================================================
# DICIONÁRIO DE CLASSIFICAÇÃO
# Mapeie os valores do seu raster CLASS para nomes e cor de display.
# Adicione ou remova linhas conforme seu projeto.
# Formato:  valor_raster : (nome_legível, código_LAS_padrão)
# ================================================================

CLASS_DICT = {
    0 : ("Não classificado",  0),
    1 : ("Solo / Terreno",    2),   # LAS standard: Ground
    2 : ("Vegetação baixa",   3),   # LAS standard: Low Vegetation
    3 : ("Vegetação média",   4),   # LAS standard: Medium Vegetation
    4 : ("Vegetação alta",    5),   # LAS standard: High Vegetation
    5 : ("Edificação",        6),   # LAS standard: Building
    6 : ("Água",             9),    # LAS standard: Water
    # Adicione mais conforme necessário:
    # 7 : ("Estrada",         11),
    # 8 : ("Ruído",            7),
}

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

def create_header(src_ortho, z_offset: float = 0.0):
    header = laspy.LasHeader(point_format=7, version="1.4")

    header.scales  = np.array([SCALE_XY, SCALE_XY, SCALE_Z])
    header.offsets = np.array([
        src_ortho.bounds.left,
        src_ortho.bounds.bottom,
        z_offset,
    ])

    # ── CRS: detecta EPSG automaticamente do ortomosaico ───────
    crs_ok = False

    if src_ortho.crs:
        try:
            from pyproj import CRS as ProjCRS
            proj_crs = ProjCRS.from_user_input(src_ortho.crs)
            epsg = proj_crs.to_epsg()

            if epsg:
                # Usa EPSG numérico direto — mais compatível com Metashape
                crs_final = ProjCRS.from_epsg(epsg)
                header.add_crs(crs_final)
                print(f"CRS detectado      : EPSG:{epsg}  ({proj_crs.name})")
                crs_ok = True
            else:
                # EPSG não mapeável, grava WKT direto
                header.add_crs(proj_crs)
                print(f"CRS detectado      : {proj_crs.name}  (sem EPSG, gravado como WKT)")
                crs_ok = True

        except Exception as e:
            print(f"[AVISO] Falha ao detectar CRS: {e}")

    if not crs_ok:
        print("[AVISO] CRS NÃO gravado no header.")
        print("        Dica: verifique se o ortomosaico possui projeção definida.")

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
# VERIFICAÇÃO DO ARQUIVO GERADO
# ================================================================

def verify_output(path: str, class_counts: dict):
    """
    Abre o LAS/LAZ gerado, verifica bounding box, projeção e
    imprime o dicionário de classificação com contagem de pontos.
    """
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DO ARQUIVO GERADO")
    print("=" * 60)

    try:
        las = laspy.read(path)
    except Exception as e:
        print(f"  [ERRO] Não foi possível abrir o arquivo: {e}")
        return

    # ----------------------------------------------------------
    # Bounding Box real dos pontos
    # ----------------------------------------------------------
    min_x, max_x = float(las.x.min()), float(las.x.max())
    min_y, max_y = float(las.y.min()), float(las.y.max())
    min_z, max_z = float(las.z.min()), float(las.z.max())

    dx = max_x - min_x
    dy = max_y - min_y
    dz = max_z - min_z

    print(f"\n[BOUNDING BOX DOS PONTOS]")
    print(f"  X  :  {min_x:.3f}  →  {max_x:.3f}   (ΔX = {dx:.3f} m)")
    print(f"  Y  :  {min_y:.3f}  →  {max_y:.3f}   (ΔY = {dy:.3f} m)")
    print(f"  Z  :  {min_z:.3f}  →  {max_z:.3f}   (ΔZ = {dz:.3f} m)")
    print(f"\n  Área coberta  : {dx/1000:.3f} km  x  {dy/1000:.3f} km")
    print(f"  Área total    : {(dx*dy)/1e6:.4f} km²")

    # ----------------------------------------------------------
    # Projeção / CRS
    # ----------------------------------------------------------
    print(f"\n[PROJEÇÃO / CRS]")
    crs_found = False

    try:
        vlrs = las.header.vlrs
        for vlr in vlrs:
            desc = str(vlr.description).strip()
            user = str(vlr.user_id).strip()
            if "WKT" in desc.upper() or "WKT" in user.upper() or vlr.record_id in (2112,):
                raw = vlr.record_data
                if isinstance(raw, (bytes, bytearray)):
                    wkt = raw.decode("utf-8", errors="ignore").strip().rstrip("\x00")
                else:
                    wkt = str(raw)
                if wkt:
                    # Tenta extrair só o nome do CRS da string WKT
                    import re
                    m = re.search(r'PROJCS\["([^"]+)"', wkt)
                    if not m:
                        m = re.search(r'GEOGCS\["([^"]+)"', wkt)
                    if m:
                        print(f"  CRS    : {m.group(1)}")
                    else:
                        print(f"  CRS    : (WKT presente, {len(wkt)} chars)")
                    print(f"  WKT    : {wkt[:120]}{'...' if len(wkt)>120 else ''}")
                    crs_found = True
                    break
    except Exception:
        pass

    if not crs_found:
        # Tenta via pyproj se disponível
        try:
            from pyproj import CRS as ProjCRS
            crs = ProjCRS.from_user_input(las.header.parse_crs())
            print(f"  CRS    : {crs.name}")
            print(f"  EPSG   : {crs.to_epsg()}")
            crs_found = True
        except Exception:
            pass

    if not crs_found:
        print("  CRS    : ⚠  Não encontrado no arquivo.")
        print("           Dica: verifique se o CRS foi gravado no header.")
        print("           Para forçar, use PDAL:")
        print(f"           pdal translate {path} {path} --writers.las.a_srs=\"EPSG:XXXX\"")

    # ----------------------------------------------------------
    # Dicionário de classificação com contagem real
    # ----------------------------------------------------------
    print(f"\n[DICIONÁRIO DE CLASSIFICAÇÃO]")
    print(f"  {'Valor':>6}  {'Rótulo':<22}  {'Cód.LAS':>7}  {'Pontos':>14}  {'%':>6}")
    print(f"  {'-'*6}  {'-'*22}  {'-'*7}  {'-'*14}  {'-'*6}")

    total_pts = sum(class_counts.values()) if class_counts else 0

    # Coleta valores presentes no arquivo
    try:
        cls_array = las.classification
        unique_vals, counts = np.unique(cls_array, return_counts=True)
        real_counts = dict(zip(unique_vals.tolist(), counts.tolist()))
    except Exception:
        real_counts = class_counts  # fallback para contagem em memória

    all_vals = sorted(
        set(real_counts.keys()) | set(CLASS_DICT.keys())
    )

    for val in all_vals:
        cnt   = real_counts.get(val, 0)
        pct   = (cnt / total_pts * 100) if total_pts > 0 else 0.0
        label, las_code = CLASS_DICT.get(val, (f"Classe {val}", val))
        marker = "  ●" if cnt > 0 else "  ○"
        print(f"{marker} {val:>5}  {label:<22}  {las_code:>7}  {cnt:>14,}  {pct:>5.1f}%")

    print()
    print(f"  Total de pontos classificados : {total_pts:,}")

    # ----------------------------------------------------------
    # Info geral do header
    # ----------------------------------------------------------
    print(f"\n[HEADER]")
    print(f"  Point Format : {las.header.point_format.id}")
    print(f"  LAS Version  : {las.header.version}")
    print(f"  N° de pontos : {len(las.x):,}")
    print(f"  Scale XYZ    : {las.header.scales}")
    print(f"  Offset XYZ   : {las.header.offsets}")

    print("\n" + "=" * 60)


# ================================================================
# DETECTA TIPO RGB E NORMALIZA PARA uint16 (0–65535)
# Suporta: uint8 (0-255), uint16 (0-65535), float32 (0.0-1.0)
# ================================================================

def detect_rgb_scale(src_ortho) -> str:
    """Retorna 'uint8', 'uint16' ou 'float' conforme dtype do ortomosaico."""
    dt = src_ortho.dtypes[0]
    if "float" in dt:
        return "float"
    if dt == "uint16":
        return "uint16"
    return "uint8"   # default


def to_uint16(band: np.ndarray, mode: str) -> np.ndarray:
    """Converte banda para uint16 (escala LAS 16-bit)."""
    if mode == "float":
        # float 0.0–1.0  →  0–65535
        arr = np.clip(band, 0.0, 1.0) * 65535.0
    elif mode == "uint16":
        arr = band.astype(np.float32)  # já está na escala certa
    else:
        # uint8  0–255  →  0–65535  (× 256)
        arr = band.astype(np.float32) * 256.0
    return np.clip(arr, 0, 65535).astype(np.uint16)


# ================================================================
# PROCESSAR BLOCO  (sem .sample – tudo por índice de pixel)
# ================================================================

def process_block(ortho, dsm, cls, window: Window, rgb_mode: str = "uint8"):
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

    rr_raw = r[rows, cols][valid_z]
    gg_raw = g[rows, cols][valid_z]
    bb_raw = b[rows, cols][valid_z]

    # converte para uint16 corretamente conforme tipo da banda
    rr = to_uint16(rr_raw, rgb_mode)
    gg = to_uint16(gg_raw, rgb_mode)
    bb = to_uint16(bb_raw, rgb_mode)

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
        rr,   # já uint16 correto
        gg,
        bb,
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

        # ── Detecta tipo da banda RGB ─────────────────────────
        rgb_mode = detect_rgb_scale(ortho)
        print(f"Tipo RGB detectado : {ortho.dtypes[0]}  →  modo '{rgb_mode}'")

        # ── Amostra Z mínimo do DSM para offset correto ───────
        print("Amostrando Z mínimo do DSM (necessário para offset)...")
        try:
            dsm_sample = dsm.read(1, out_shape=(256, 256), resampling=Resampling.nearest)
            nodata_v = dsm.nodata if dsm.nodata is not None else -9999.0
            dsm_valid = dsm_sample[np.isfinite(dsm_sample) & (dsm_sample != nodata_v)]
            z_offset = float(np.min(dsm_valid)) if len(dsm_valid) > 0 else 0.0
        except Exception:
            z_offset = 0.0
        print(f"Offset Z           : {z_offset:.3f} m")

        W, H = ortho.width, ortho.height
        total_px = W * H

        header = create_header(ortho, z_offset=z_offset)

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
        class_counts    = {}          # contagem de pontos por classe

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
                        result = process_block(ortho, dsm, cls, window, rgb_mode)
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
                    pts.red            = rs   # já uint16 (to_uint16 aplicado)
                    pts.green          = gs
                    pts.blue           = bs
                    pts.classification = cls_vals

                    writer.write_points(pts)

                    # ----------------------------------------
                    # Estatísticas
                    # ----------------------------------------
                    total_written += n
                    global_classes.update(np.unique(cls_vals).tolist())

                    # acumula contagem por classe
                    for v, c in zip(*np.unique(cls_vals, return_counts=True)):
                        class_counts[int(v)] = class_counts.get(int(v), 0) + int(c)

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
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)
    print(f"  Arquivo        : {OUTPUT}")
    print(f"  Pontos totais  : {total_written:,}")
    print(f"  Tempo total    : {total_time/60:.2f} min")
    print(f"  Vel. média     : {total_written/total_time:,.0f} pts/s")
    print(f"  Z MIN          : {min_z:.3f}")
    print(f"  Z MAX          : {max_z:.3f}")
    print()
    print("  Atributos      : X  Y  Z  Red  Green  Blue  Classification")
    print("  Point Format   : 7  |  LAS 1.4")
    print("  Compatível com : LiDAR360  CloudCompare  PDAL  QGIS  ArcGIS Pro")

    # ================================================================
    # VERIFICAÇÃO PÓS-ESCRITA
    # ================================================================
    verify_output(OUTPUT, class_counts)


# ================================================================

if __name__ == "__main__":
    main()