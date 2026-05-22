# Raster → LAS/LAZ Colorido com Altura e Classificação

Converte ortomosaicos RGB, modelos digitais de superfície (DSM/MDS) e rasters de classificação em nuvens de pontos **LAS/LAZ** no formato padrão **LAS 1.4 / Point Format 7** (X, Y, Z, Red, Green, Blue, Classification).

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Visão Geral

Este script transforma **três rasters georreferenciados** em uma única nuvem de pontos LAS/LAZ:

| Entrada          | Descrição                                              |
|------------------|--------------------------------------------------------|
| **Ortomosaico**  | RGB (3 bandas) — uint8, uint16 ou float                |
| **DSM / MDS**    | Modelo Digital de Superfície (altura) — 1 banda        |
| **Classificação**| Raster temático (valores 0, 1, 2, 3...) — 1 banda      |

A saída é um arquivo **LAS 1.4** contendo coordenadas geográficas (X, Y), altitude (Z), cor RGB (16 bits) e classe LAS para cada pixel do ortomosaico.

---

## Funcionalidades

- ✅ **Processamento em blocos** — memória controlada, ideal para grandes áreas
- ✅ **REAMOSTRAGEM automática** — DSM e classificação são reamostrados para a grade do ortomosaico (suporta resoluções diferentes entre os rasters)
- ✅ **CRS automático** — detecta o sistema de referência do ortomosaico e grava no header LAS (EPSG ou WKT)
- ✅ **RGB inteligente** — suporta ortomosaicos uint8, uint16 e float32 (normaliza para uint16 conforme padrão LAS)
- ✅ **Classificação LAS** — mapeia valores do raster de classificação para códigos padrão LAS (Ground, Vegetation, Building, Water…)
- ✅ **Filtro de pixels pretos** — opção para ignorar pixels RGB (0, 0, 0) opcional
- ✅ **Verificação pós-escrita** — exibe bounding box, CRS, estatísticas por classe e informações do header
- ✅ **Barra de progresso** — logs detalhados com velocidade, ETA e classes encontradas por bloco
- ✅ **LAZ nativo** — saída compactada (extensão .laz) com a biblioteca `lazrs`

---

## Instalação

```bash
pip install numpy rasterio laspy lazrs
```

> `laspy` + `lazrs` fornecem suporte a escrita LAZ.  
> `rasterio` gerencia a leitura dos rasters em blocos.  
> `numpy` processa os arrays vetorizadamente (sem loops lentos).

---

## Uso

### 1. Configuração

Edite o início do script com os caminhos dos seus arquivos:

```python
ORTHO  = r"C:\projeto\ortomosaico.tif"
DSM    = r"C:\projeto\dsm.tif"
CLASS  = r"C:\projeto\classificacao.tif"
OUTPUT = r"C:\projeto\nuvem.las"
```

### 2. Ajustes opcionais

```python
BLOCK_SIZE = 8192      # pixels por bloco (reduza se faltar RAM)
SCALE_XY   = 0.001     # precisão XY em metros
SCALE_Z    = 0.001     # precisão Z em metros
SKIP_BLACK = False     # True → ignora pixels RGB (0,0,0)
DEBUG      = True      # logs detalhados dos blocos
```

### 3. Dicionário de classificação

Mapeie os valores do seu raster de classificação para os códigos LAS padrão:

```python
CLASS_DICT = {
    0 : ("Não classificado",  0),
    1 : ("Solo / Terreno",    2),   # Ground
    2 : ("Vegetação baixa",   3),   # Low Vegetation
    3 : ("Vegetação média",   4),   # Medium Vegetation
    4 : ("Vegetação alta",    5),   # High Vegetation
    5 : ("Edificação",        6),   # Building
    6 : ("Água",             9),   # Water
}
```

### 4. Execute

```bash
python RTL6_faltaepsg.py
```

---

## Exemplo de saída

```
====================================================================
PROCESSAMENTO CONCLUÍDO
====================================================================
  Arquivo        : D:\TESTES_PYTHON\RASTER_TO_LAS\IMARU_teste_rtl6.las
  Pontos totais  : 45.829.440
  Tempo total    : 12.34 min
  Vel. média     : 61.800 pts/s
  Z MIN          : 712.345
  Z MAX          : 891.234

  Atributos      : X  Y  Z  Red  Green  Blue  Classification
  Point Format   : 7  |  LAS 1.4
  Compatível com : LiDAR360  CloudCompare  PDAL  QGIS  ArcGIS Pro
```

---

## Requisitos dos Rasters

| Requisito             | Detalhe                                                |
|-----------------------|--------------------------------------------------------|
| **Mesmo CRS**         | ORTHO, DSM e CLASS devem estar no mesmo sistema de projeção |
| **Sobreposição**      | As áreas precisam se sobrepor geograficamente           |
| **Bandas**            | ORTHO = 3 bandas (RGB); DSM e CLASS = 1 banda cada     |
| **Nodata**            | DSM e CLASS podem ter nodata (valores ignorados)        |
| **Tipos numéricos**   | ORTHO: uint8, uint16 ou float32;<br>DSM: float (altura em metros);<br>CLASS: inteiro |

---

## Arquitetura do Processamento

```
Ortomosaico (grande)
     │
     ▼
 ┌──────────────────────────────┐
 │  Divisão em blocos (8192²)   │
 └──────┬───────────────────────┘
        │
        ▼
 ┌──────────────────────────────┐
 │  Leitura RGB do ORTHO        │
 │  + Window DSM/CLASS          │
 │  + Reamostragem → grade ORTHO│
 └──────┬───────────────────────┘
        │
        ▼
 ┌──────────────────────────────┐
 │  Máscara de pixels válidos   │
 │  Cálculo de coordenadas      │
 │  (vetorizado com numpy)      │
 └──────┬───────────────────────┘
        │
        ▼
 ┌──────────────────────────────┐
 │  Normalização RGB → uint16   │
 │  Class → código LAS (uint8)  │
 └──────┬───────────────────────┘
        │
        ▼
 ┌──────────────────────────────┐
 │  Escrita no LAS (bloco a     │
 │  bloco via laspy)            │
 └──────────────────────────────┘
```

---

## Parâmetros detalhados

| Parâmetro    | Default | Descrição                                              |
|-------------|---------|--------------------------------------------------------|
| `ORTHO`     | —       | Caminho do ortomosaico RGB (.tif)                      |
| `DSM`       | —       | Caminho do Modelo Digital de Superfície (.tif)         |
| `CLASS`     | —       | Caminho do raster de classificação (.tif)              |
| `OUTPUT`    | —       | Caminho de saída (.las ou .laz)                        |
| `BLOCK_SIZE`| 8192    | Tamanho do bloco em pixels (altura e largura)          |
| `SCALE_XY`  | 0.001   | Precisão das coordenadas X e Y em metros               |
| `SCALE_Z`   | 0.001   | Precisão da altitude Z em metros                       |
| `SKIP_BLACK`| False   | Ignora pixels com RGB = (0,0,0)                        |
| `DEBUG`     | True    | Exibe logs detalhados por bloco processado             |

---

## Compatibilidade

A nuvem de pontos gerada pode ser aberta em:

- **CloudCompare** (importar como LAS/LAZ)
- **PDAL** (pipeline de processamento)
- **QGIS** (plugin LAS)
- **ArcGIS Pro** (dados LAS)
- **LiDAR360**
- **Global Mapper**
- **LASTools**

---

## Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais informações.

---

## Contribuição

Sugestões, issues e pull requests são bem-vindos!

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/minha-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/minha-feature`)
5. Abra um Pull Request

---

## Autor

Desenvolvido por [MartinelliNTZ](https://github.com/MartinelliNTZ)

Repositório: [RastesToLasTool](https://github.com/MartinelliNTZ/RastesToLasTool.git)