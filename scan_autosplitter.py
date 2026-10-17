#!/usr/bin/env python3
"""
scan_autosplitter.py — automatyczne wykrywanie, wycinanie i prostowanie zdjęć ze skanów.

Pipeline (klasyczne CV, OpenCV + NumPy):
    1. Wczytanie skanu z zachowaniem głębi bitowej (8/16 bit, TIFF/PNG/JPG).
    2. Detekcja na pomniejszonej kopii roboczej:
         - kolor tła skanera = dominujący jasny, neutralny odcień (zdjęcia mogą sięgać brzegów),
         - maska "odległości od tła" (ΔE w Lab) + krawędzie Canny,
         - usunięcie ramki/cienia skanera (linie przy brzegu pokryte prawie na całej długości),
         - małe domknięcie, wypełnienie dziur, otwarcie (usuwa kurz, włosy, cienkie mostki),
         - rozdzielanie sklejonych zdjęć (transformata odległości + watershed),
         - cięcie wzdłuż wąskich, prostych szczelin (profile linii pod kątami ±3°),
         - filtracja konturów: pole, wymiary, prostokątność.
    3. Doprecyzowanie każdego prostokąta w wyższej rozdzielczości (minAreaRect otoczki wypukłej)
       i rozszerzenie o jasny margines odbitki (biała ramka) na podstawie profili linii.
    4. Deskew: transformacja perspektywiczna prostokąta obróconego -> prostokąt osiowy
       (Lanczos), opcjonalne przycięcie resztek tła na brzegach. Kąt ~0° = czyste wycięcie,
       piksele identyczne bit w bit z oryginałem.
    5. Orientacja: obrót o 90/180/270° tak, by twarze stały pionowo (detektor YuNet, lokalnie).
    6. Zapis bezstratny (PNG / nieskompresowany TIFF) jako <skan>_crop_NN.<ext>.

Użycie:
    python scan_autosplitter.py WEJSCIE WYJSCIE [--format png|tiff] [--workers N] [--debug]
                                [--no-face-rotate]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as _exc:  # najczęstszy problem przy pierwszym uruchomieniu
    sys.exit(
        f"Brak biblioteki: {_exc.name}\n"
        f"Skrypt uruchomiono interpreterem: {sys.executable}\n"
        f"Zainstaluj biblioteki DLA TEGO interpretera:\n"
        f'    "{sys.executable}" -m pip install -r requirements.txt\n'
        f"(Jeśli to nie jest Python, którego chcesz używać — zobacz README, sekcja 6.)"
    )

# =============================================================================
# KONFIGURACJA — główne parametry do strojenia
# =============================================================================

# --- Wejście / wyjście -------------------------------------------------------
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
OUTPUT_FORMAT = "png"            # "png" (bezstratny, kompresja Deflate) lub "tiff" (bez kompresji)
PNG_COMPRESSION_LEVEL = 3        # 0-9; wpływa wyłącznie na rozmiar pliku i czas, NIE na jakość
CROP_NAME_PATTERN = "{stem}_crop_{index:02d}.{ext}"
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# --- Detekcja (na pomniejszonej kopii roboczej) ------------------------------
# Rozmiary jąder morfologicznych są ułamkiem DŁUŻSZEGO BOKU SKANU, więc nie zależą od DPI.
# Dla A4 (297 mm): 0.001 ≈ 0.3 mm.
WORK_MAX_SIDE = 2000             # dłuższy bok kopii roboczej [px]; większy = dokładniej, wolniej
BG_MAX_CHROMA = 12.0             # max. "kolorowość" piksela tła (tło skanera jest neutralne: biel/szarość)
BG_DIFF_THRESHOLD = 14.0         # min. odległość koloru od tła (ΔE w Lab, skala 0-255) = "zdjęcie"
BG_DIFF_BLUR = 5                 # rozmycie mapy różnic (tłumi szum/ziarno skanera), nieparzyste
CANNY_LOW = 30                   # dolny próg Canny (krawędzie jasnych zdjęć na jasnym tle)
CANNY_HIGH = 90                  # górny próg Canny
USE_EDGES = True                 # łącz maskę koloru z krawędziami (pomaga przy jasnych zdjęciach)
CLOSE_KERNEL_FRAC = 0.004        # domknięcie (~1.2 mm na A4) — łata krawędzie zdjęć; MUSI być
                                 # mniejsze niż najwęższa przerwa między zdjęciami
OPEN_KERNEL_FRAC = 0.006         # otwarcie (~1.8 mm na A4) — usuwa kurz, włosy i cienkie "mostki"
SCANNER_EDGE_LINE_FRAC = 0.95    # wiersz/kolumna przy brzegu skanu pokryta "nie-tłem" w >= tyle
                                 # długości to ramka/cień skanera (zdjęcia nie pokrywają całego boku)
SCANNER_EDGE_MAX_FRAC = 0.01     # max. głębokość usuwanej ramki skanera (ułamek boku)

# --- Filtracja kandydatów ----------------------------------------------------
MIN_AREA_FRAC = 0.008            # min. pole zdjęcia jako ułamek pola skanu (odrzuca śmieci)
MAX_AREA_FRAC = 0.98             # max. pole (odrzuca "cały skan" jako jedno zdjęcie)
MIN_SIDE_FRAC = 0.04             # min. długość krótszego boku jako ułamek dłuższego boku skanu
MAX_ASPECT_RATIO = 8.0           # max. proporcje boków (odrzuca paski/listwy skanera)
MIN_RECTANGULARITY = 0.85        # min. pole konturu / pole minAreaRect dla części po rozdzieleniu
SPLIT_CORE_LEVELS = (0.5, 0.6, 0.7, 0.8)  # progi transformaty odległości (ułamek maks.) do szukania "jąder" zdjęć
SPLIT_MIN_PART_FRAC = 0.12       # część po rozdzieleniu musi mieć >= tyle pola całego obszaru
SPLIT_WORK_SIDE = 400            # rozdzielanie liczone na obszarze pomniejszonym do tylu px (szybkość)

# --- Cięcie wzdłuż wąskich przerw (zdjęcia ułożone równo, prawie stykające się) ---
GAP_SPLIT = True                 # szukaj prostych, wąskich linii tła przecinających obszar
GAP_BG_DIFF = 22.0               # piksel przerwy: ΔE od tła skanera <= tyle
GAP_COUNT_PAPER_WHITE = True     # licz też jasny neutralny piksel (biały brzeg odbitki) jako przerwę
GAP_MIN_FRAC = 0.55              # linia cięcia: >= tyle jej długości to przerwa (reszta: zdjęcia się stykają)
GAP_SIDE_MAX_FRAC = 0.3          # linie kontrolne po obu stronach cięcia mogą mieć najwyżej tyle "przerwy"
                                 # — odróżnia wąską szczelinę od jasnego fragmentu zdjęcia
GAP_STRONG_FRAC = 0.9            # wyjątek: linia przerwy prawie na całej długości (>= tyle) może mieć
GAP_CONTRAST = 0.12              # jasną treść (np. prześwietlone niebo) po jednej stronie, o ile ta
                                 # strona ma o tyle mniej "przerwy" niż samo cięcie
GAP_BAND_FRAC = 0.0012           # połowa szerokości pasa wokół linii (ułamek boku skanu; ~0.35 mm na A4)
GAP_SIDE_OFFSET_FRAC = 0.008     # odległość linii kontrolnych od cięcia (~2.4 mm na A4)
GAP_MAX_ANGLE = 3.0              # max. odchylenie linii cięcia od boku obszaru [°] (krzywo leżące zdjęcia)
GAP_ANGLE_STEP = 0.25            # krok przeszukiwania kąta [°]
GAP_MIN_PART_FRAC = 0.15         # każda część po cięciu musi mieć >= tyle szerokości całego obszaru
GAP_MAX_CONTINUITY = 0.45        # max. korelacja treści po obu stronach cięcia; przy prawdziwej szczelinie
                                 # sąsiadują dwa różne zdjęcia (korelacja ~0), a jasny element zdjęcia
                                 # (np. biały filar budynku) ma po obu stronach tę samą strukturę
GAP_PART_MAX_ASPECT = 3.2        # części po cięciu muszą wyglądać jak odbitki: proporcje boków <= tyle
GAP_PART_MIN_SIDE_FRAC = 0.08    # ...i krótszy bok >= tyle dłuższego boku skanu (~24 mm na A4)
GAP_MAX_DEPTH = 6                # max. głębokość rekurencji (2^6 = 64 zdjęcia w jednym obszarze)

# --- Doprecyzowanie i prostowanie --------------------------------------------
REFINE_FULL_RES = True           # doprecyzuj krawędzie w wyższej rozdzielczości
REFINE_MAX_SIDE = 6000           # doprecyzowanie na kopii o dłuższym boku <= tyle px (~500 dpi dla A4);
                                 # skany 1200 dpi nie potrzebują większej precyzji, a zużywają dużo RAM
REFINE_MARGIN_FRAC = 0.03        # margines ROI przy doprecyzowaniu (ułamek boku zdjęcia)
REFINE_OPEN_FRAC = 0.02          # otwarcie przy doprecyzowaniu (ułamek boku zdjęcia) — odcina kurz
                                 # przyklejony do krawędzi, który zawyżałby otoczkę wypukłą
GROW_TO_PAPER_EDGE = True        # rozszerz ramkę o jasny margines odbitki (biała ramka zdjęcia),
                                 # który jest za mało kontrastowy dla maski — wykrywany profilami linii
GROW_MAX_FRAC = 0.06             # max. rozszerzenie z każdej strony (ułamek boku zdjęcia)
GROW_DELTA = 3.0                 # linia należy do odbitki, gdy jej średnie ΔE > tło lokalne + tyle
REFINE_MAX_AREA_DEVIATION = 0.12 # gdy doprecyzowany prostokąt różni się bardziej -> użyj zgrubnego
CROP_INSET_PX = 2                # stałe przycięcie do wewnątrz [px pełnej rozdz.] (anty-ramka)
AUTO_TRIM_BORDERS = True         # dodatkowo przytnij wiersze/kolumny brzegowe wyglądające jak tło
AUTO_TRIM_MAX_FRAC = 0.005       # max. przycięcie automatyczne z każdej strony (ułamek boku); mało,
                                 # bo biała ramka odbitki wygląda jak tło i nie może zostać ścięta
AUTO_TRIM_BG_DIFF = 5.0          # piksel = tło przy przycinaniu, gdy ΔE od tła <= tyle (ostrzej niż
                                 # detekcja, by nie ścinać białej ramki odbitki)
AUTO_TRIM_BG_RATIO = 0.6         # linia jest "tłem", gdy >= tyle jej pikseli jest podobnych do tła
SKIP_ROTATION_BELOW_DEG = 0.05   # poniżej tego kąta tylko wycinamy (bit-w-bit identyczne piksele)
INTERPOLATION = cv2.INTER_LANCZOS4

# --- Automatyczna orientacja wg twarzy (lokalnie, model YuNet ~230 KB) --------
AUTO_ROTATE_FACES = True         # obracaj zdjęcia o 90/180/270°, aby twarze były pionowo
FACE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
FACE_DETECT_SIDES = (640, 1024, 1600)  # rozmiary (dłuższy bok) do szukania twarzy — kolejno; większy
                                 # tylko gdy na mniejszym brak pewnej twarzy (małe twarze, zdjęcia grupowe)
FACE_SCORE_THRESHOLD = 0.6       # min. pewność pojedynczej twarzy branej pod uwagę
FACE_ENHANCE = True              # wyrównanie kontrastu (CLAHE) przed detekcją — ciemne, stare odbitki
FACE_MIN_CONFIDENCE = 0.8        # obracamy tylko, gdy w nowej orientacji jest twarz o takiej pewności
FACE_ROTATION_MARGIN = 0.1       # ...i jest ona pewniejsza niż najlepsza twarz w obecnej orientacji o tyle

# --- Kolejność wyników -------------------------------------------------------
ROW_GROUPING_FRAC = 0.5          # zdjęcia, których środki różnią się w pionie o < tyle*wysokość, są w jednym rzędzie

# =============================================================================

log = logging.getLogger("autosplitter")


@dataclass
class Detection:
    """Obrócony prostokąt w układzie pełnej rozdzielczości skanu."""
    center: tuple[float, float]
    size: tuple[float, float]
    angle: float

    @property
    def rect(self):
        return (self.center, self.size, self.angle)


# -----------------------------------------------------------------------------
# I/O (bezpieczne dla ścieżek z polskimi znakami w Windows)
# -----------------------------------------------------------------------------

def read_image(path: Path) -> np.ndarray | None:
    """Wczytuje obraz bez utraty głębi bitowej i kanałów."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:  # kanał alfa nie ma sensu dla skanu — odrzucamy
        img = img[:, :, :3]
    return img


def write_image(path: Path, img: np.ndarray, fmt: str) -> None:
    """Zapis bezstratny: PNG (Deflate) lub TIFF bez kompresji."""
    if fmt == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION_LEVEL]
        ext = ".png"
    else:
        params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]  # 1 = COMPRESSION_NONE
        ext = ".tif"
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"Nie udało się zakodować {path}")
    buf.tofile(str(path))


def to_bgr8(img: np.ndarray) -> np.ndarray:
    """Kopia 8-bit BGR wyłącznie do celów detekcji (oryginał pozostaje nietknięty)."""
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# -----------------------------------------------------------------------------
# Maska pierwszego planu
# -----------------------------------------------------------------------------

def odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def estimate_background(lab: np.ndarray) -> np.ndarray:
    """
    Kolor tła skanera = dominujący jasny i neutralny (bezbarwny) odcień na całym skanie.
    Nie zakłada, że brzegi skanu są wolne — zdjęcia mogą dochodzić do krawędzi i rogów.
    """
    px = lab[::2, ::2].reshape(-1, 3)
    chroma = np.hypot(px[:, 1] - 128.0, px[:, 2] - 128.0)
    cand = px[(chroma <= BG_MAX_CHROMA) & (px[:, 0] >= 100)]
    if len(cand) < 0.01 * len(px):
        # Prawie brak jasnych neutralnych pikseli — awaryjnie mediana z pasa brzegowego.
        h, w = lab.shape[:2]
        b = max(2, int(min(h, w) * 0.03))
        cand = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                               lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
        return np.median(cand, axis=0).astype(np.float32)
    hist = np.bincount(cand[:, 0].astype(np.int32), minlength=256).astype(np.float32)
    hist = np.convolve(hist, [1, 4, 6, 4, 1], mode="same")  # wygładzenie histogramu jasności
    peak = int(np.argmax(hist))
    return np.median(cand[np.abs(cand[:, 0] - peak) <= 4], axis=0).astype(np.float32)


def background_distance(bgr8: np.ndarray, bg_lab: np.ndarray | None = None):
    """Mapa ΔE (Lab) każdego piksela od koloru tła."""
    lab = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB).astype(np.float32)
    if bg_lab is None:
        bg_lab = estimate_background(lab)
    diff = np.sqrt(((lab - bg_lab) ** 2).sum(axis=2))
    k = odd(BG_DIFF_BLUR)
    if k > 1:
        diff = cv2.GaussianBlur(diff, (k, k), 0)
    return diff, bg_lab


def fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def edge_map(bgr8: np.ndarray) -> np.ndarray:
    """Krawędzie Canny — wychwytują jasne zdjęcia (np. z białą ramką) na jasnym tle."""
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 7)  # wygładza ziarno, zachowuje krawędzie
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)


def fit_rect(contour: np.ndarray):
    """minAreaRect na otoczce wypukłej — odbitki są wypukłe, więc "zatoki" w masce
    (jasne fragmenty zdjęcia przy krawędzi) nie zaniżają wymiarów prostokąta."""
    return cv2.minAreaRect(cv2.convexHull(contour))


def scanner_edge_widths(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Głębokość ramki / cienia skanera (góra, dół, lewo, prawo) w pikselach maski.
    Ramka skanera to wiersze/kolumny przy brzegu prawie w całości "nie-tłem". Zdjęcie dosunięte
    do krawędzi pokrywa tylko część boku, więc nie jest usuwane.
    """
    fg = mask > 0
    h, w = fg.shape
    def depth(lines):
        d = 0
        for line in lines:
            if line.mean() < SCANNER_EDGE_LINE_FRAC:
                break
            d += 1
        return d + 2 if d else 0  # +2 px na rozmyte przejście ramka -> tło
    my, mx = int(h * SCANNER_EDGE_MAX_FRAC) + 1, int(w * SCANNER_EDGE_MAX_FRAC) + 1
    return (depth(fg[i] for i in range(my)), depth(fg[h - 1 - i] for i in range(my)),
            depth(fg[:, i] for i in range(mx)), depth(fg[:, w - 1 - i] for i in range(mx)))


def clear_scanner_edges(mask: np.ndarray, widths, offset: tuple[int, int] = (0, 0),
                        full_shape: tuple[int, int] | None = None) -> None:
    """Zeruje (in-place) ramkę skanera. Dla ROI podajemy jego przesunięcie i wymiary pełnego skanu."""
    top, bottom, left, right = widths
    h, w = mask.shape
    H, W = full_shape or (h, w)
    x0, y0 = offset
    mask[:max(0, top - y0), :] = 0
    mask[max(0, H - bottom - y0):, :] = 0
    mask[:, :max(0, left - x0)] = 0
    mask[:, max(0, W - right - x0):] = 0


def morph(mask: np.ndarray, op: int, size: int) -> np.ndarray:
    k = odd(size)
    return cv2.morphologyEx(mask, op, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))) if k > 1 else mask


def build_foreground_mask(bgr8: np.ndarray):
    """Binarna maska obszarów będących zdjęciami (na kopii roboczej) + kolor tła + ramka skanera."""
    diff, bg_lab = background_distance(bgr8)
    mask = (diff > BG_DIFF_THRESHOLD).astype(np.uint8) * 255
    if USE_EDGES:
        mask = cv2.bitwise_or(mask, edge_map(bgr8))
    edges = scanner_edge_widths(mask)
    clear_scanner_edges(mask, edges)

    long_side = max(mask.shape)
    mask = morph(mask, cv2.MORPH_CLOSE, long_side * CLOSE_KERNEL_FRAC)
    mask = fill_holes(mask)
    mask = morph(mask, cv2.MORPH_OPEN, long_side * OPEN_KERNEL_FRAC)

    # Mapa "przerwy": tło skanera albo jasny, neutralny papier (krawędź odbitki, prześwit szczeliny).
    lab = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB)
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    gap = diff <= GAP_BG_DIFF
    if GAP_COUNT_PAPER_WHITE:
        gap |= (lab[..., 0] >= bg_lab[0] - 3) & (chroma <= BG_MAX_CHROMA)
    return mask, bg_lab, edges, gap.astype(np.uint8)


# -----------------------------------------------------------------------------
# Kandydaci
# -----------------------------------------------------------------------------

def rect_ok(rect, img_area: float, img_long: float) -> bool:
    (_, _), (w, h), _ = rect
    if w <= 0 or h <= 0:
        return False
    area = w * h
    short, long_ = min(w, h), max(w, h)
    return (MIN_AREA_FRAC * img_area <= area <= MAX_AREA_FRAC * img_area
            and short >= MIN_SIDE_FRAC * img_long
            and long_ / short <= MAX_ASPECT_RATIO)


def rectangularity(contour: np.ndarray, rect) -> float:
    rw, rh = rect[1]
    return cv2.contourArea(contour) / max(rw * rh, 1e-6)


def flood_from_cores(cores: np.ndarray, dist: np.ndarray, steps: int = 24) -> np.ndarray:
    """
    Watershed po mapie odległości: jądra rosną poziomami od najwyższej wartości `dist`
    w dół, więc spotykają się dokładnie w przewężeniu (miejscu styku zdjęć).
    (cv2.watershed zalewa wg gradientu koloru, a nie wysokości reliefu — tu się nie nadaje.)
    """
    labels = cores.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for t in np.linspace(dist.max(), 0.0, steps + 1)[1:]:
        allowed = dist > t if t > 0 else dist > 0
        while True:
            grown = cv2.dilate(labels, kernel)
            fill = (labels == 0) & allowed & (grown > 0)
            if not fill.any():
                break
            labels[fill] = grown[fill]
    return labels.astype(np.int32)


def split_touching(contour: np.ndarray, img_area: float, img_long: float) -> list:
    """
    Rozdziela stykające się / sklejone zdjęcia (transformata odległości + watershed).

    W transformacie odległości każda odbitka tworzy własne "jądro" (maksimum w środku),
    a miejsce styku to wąskie gardło o niższej wartości. Progujemy mapę na kilku poziomach;
    jeśli powstaje >=2 jąder, watershed rozrasta je z powrotem do granic maski, tnąc obszar
    dokładnie w przewężeniu. Pojedyncza wypukła odbitka ma jedno jądro, więc nie jest dzielona.
    Liczone na wycinku obszaru pomniejszonym do SPLIT_WORK_SIDE px.
    """
    x, y, w, h = cv2.boundingRect(contour)
    comp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.drawContours(comp, [contour - (x - 1, y - 1)], -1, 255, cv2.FILLED)
    f = min(1.0, SPLIT_WORK_SIDE / max(h, w))
    small = cv2.resize(comp, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) if f < 1 else comp
    dist = cv2.distanceTransform(small, cv2.DIST_L2, 5)
    comp_area = float(cv2.countNonZero(comp))
    for level in SPLIT_CORE_LEVELS:
        n, cores = cv2.connectedComponents((dist > level * dist.max()).astype(np.uint8))
        if n - 1 < 2:
            continue
        markers = flood_from_cores(cores, dist)
        if f < 1:
            markers = cv2.resize(markers.astype(np.float32), (comp.shape[1], comp.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(np.int32)
        parts = []
        for lbl in range(1, n):
            part = ((markers == lbl) & (comp > 0)).astype(np.uint8) * 255
            if cv2.countNonZero(part) < SPLIT_MIN_PART_FRAC * comp_area:
                break
            cnts, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cnts, key=cv2.contourArea) + (x - 1, y - 1)
            r = fit_rect(c)
            if not rect_ok(r, img_area, img_long) or rectangularity(c, r) < MIN_RECTANGULARITY:
                break
            parts.append((c, r))
        else:
            return parts
    return []


def upright_frame(rect):
    """Macierz afiniczna: układ obrazu -> układ prostokąta (oś x wzdłuż górnego boku) i jego wymiary."""
    src = order_corners(cv2.boxPoints(rect))
    w, h = np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[3] - src[0])
    ux, uy = (src[1] - src[0]) / max(w, 1e-6), (src[3] - src[0]) / max(h, 1e-6)
    A = np.array([[ux[0], ux[1], -src[0] @ ux], [uy[0], uy[1], -src[0] @ uy]], np.float32)
    return A, int(np.ceil(w)) + 1, int(np.ceil(h)) + 1


def gap_cut_candidates(gap_u: np.ndarray, mask_u: np.ndarray, long_side: float) -> list:
    """
    Szuka w obszarze (w układzie wyprostowanym) prostych linii przerwy.

    Dla kątów ±GAP_MAX_ANGLE obracamy mapę przerw i liczymy, jaka część każdego wiersza
    (cięcie poziome) i kolumny (cięcie pionowe) jest przerwą. Szczelina między zdjęciami daje
    wysoki, wąski pik; jasny fragment zdjęcia — szeroki "płaskowyż" (odrzucany liniami kontrolnymi).
    Zwraca kandydatów (wynik, macierz obrotu, oś 0/1, pozycja) od najlepszego.
    """
    h, w = mask_u.shape
    b = max(1, int(round(GAP_BAND_FRAC * long_side)))
    off = max(b + 2, int(round(GAP_SIDE_OFFSET_FRAC * long_side)))
    found = []
    for theta in np.arange(-GAP_MAX_ANGLE, GAP_MAX_ANGLE + 1e-6, GAP_ANGLE_STEP):
        R = cv2.getRotationMatrix2D((w / 2, h / 2), float(theta), 1.0)
        g = cv2.warpAffine(gap_u, R, (w, h), flags=cv2.INTER_NEAREST)
        m = cv2.warpAffine(mask_u, R, (w, h), flags=cv2.INTER_NEAREST)
        for axis in (0, 1):  # 0: linie poziome (profil wierszy), 1: pionowe (profil kolumn)
            k = (1, 2 * b + 1) if axis else (2 * b + 1, 1)
            gd = cv2.dilate(g, np.ones(k, np.uint8)) & m
            den = m.sum(axis=1 - axis).astype(np.float32)
            num = gd.sum(axis=1 - axis).astype(np.float32)
            ok = den >= 0.5 * den.max()
            idx = np.flatnonzero(ok)
            if len(idx) < 3 * off:
                continue
            lo, hi = idx[0], idx[-1]
            frac = np.where(ok, num / np.maximum(den, 1), np.nan)
            min_part = max(GAP_MIN_PART_FRAC * (hi - lo), off)
            pos = np.arange(int(lo + min_part), int(hi - min_part) + 1)
            if len(pos) == 0:
                continue
            centre = np.nan_to_num(frac[pos], nan=0.0)
            left = np.nan_to_num(frac[pos - off], nan=1.0)
            right = np.nan_to_num(frac[pos + off], nan=1.0)
            lo_side, hi_side = np.minimum(left, right), np.maximum(left, right)
            good = (centre >= GAP_MIN_FRAC) & (lo_side <= GAP_SIDE_MAX_FRAC) & (
                (hi_side <= GAP_SIDE_MAX_FRAC) | ((centre >= GAP_STRONG_FRAC) & (hi_side <= centre - GAP_CONTRAST)))
            score = np.where(good, centre - hi_side, -1.0)
            # Lokalne maksima (po jednym na "pik"), najlepsze 3 dla danego kąta i osi.
            for i in np.argsort(score)[::-1][:3]:
                if score[i] <= 0:
                    break
                found.append((float(score[i]), R, axis, int(pos[i]), off))
    found.sort(key=lambda c: -c[0])
    return found


def content_continuity(gray_u: np.ndarray, mask_u: np.ndarray, R: np.ndarray, axis: int,
                       pos: int, off: int) -> float:
    """
    Korelacja treści po obu stronach linii cięcia (pasy szerokości ~off tuż za liniami
    kontrolnymi). Prawdziwa szczelina rozdziela dwa niezwiązane zdjęcia -> korelacja bliska 0.
    """
    h, w = mask_u.shape
    g = cv2.warpAffine(gray_u, R, (w, h), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(mask_u, R, (w, h), flags=cv2.INTER_NEAREST)
    if axis:
        g, m = g.T, m.T
    n = g.shape[0]
    a0, a1 = max(0, pos - 2 * off), max(0, pos - off)
    b0, b1 = min(n, pos + off + 1), min(n, pos + 2 * off + 1)
    if a1 - a0 < 2 or b1 - b0 < 2:
        return 0.0
    valid = (m[a0:a1].min(axis=0) > 0) & (m[b0:b1].min(axis=0) > 0)
    if valid.sum() < 20:
        return 0.0
    sa, sb = g[a0:a1].mean(axis=0)[valid], g[b0:b1].mean(axis=0)[valid]
    k = max(3, len(sa) // 10) | 1  # usunięcie wolnej zmienności (gradient oświetlenia, winieta)
    sa = sa - cv2.blur(sa.reshape(-1, 1), (1, k)).ravel()
    sb = sb - cv2.blur(sb.reshape(-1, 1), (1, k)).ravel()
    if sa.std() < 1.0 or sb.std() < 1.0:
        return 0.0  # po jednej stronie brak struktury (np. jednolite niebo) — nie ma czego ciągnąć
    return float(np.corrcoef(sa, sb)[0, 1])


def gap_cut_regions(region: np.ndarray, gap: np.ndarray, gray: np.ndarray, img_area: float,
                    img_long: float, depth: int = 0) -> list:
    """Rekurencyjnie tnie obszar (maskę w układzie roboczym) wzdłuż przerw. Zwraca prostokąty."""
    pts = cv2.findNonZero(region)
    if pts is None:
        return []
    rect = cv2.minAreaRect(cv2.convexHull(pts))
    if depth >= GAP_MAX_DEPTH or rect[1][0] * rect[1][1] < 2 * MIN_AREA_FRAC * img_area:
        return [rect]
    A, w, h = upright_frame(rect)
    gap_u = cv2.warpAffine(gap, A, (w, h), flags=cv2.INTER_NEAREST)
    mask_u = cv2.warpAffine(region, A, (w, h), flags=cv2.INTER_NEAREST) // 255
    gray_u = cv2.warpAffine(gray, A, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
    cut = None
    for _, R, axis, pos, off in gap_cut_candidates(gap_u, mask_u, img_long)[:8]:
        corr = content_continuity(gray_u, mask_u, R, axis, pos, off)
        if corr <= GAP_MAX_CONTINUITY:
            cut = (R, axis, pos)
            break
        log.debug("Odrzucono linię cięcia: treść ciągnie się przez nią (korelacja %.2f)", corr)
    if cut is None:
        return [rect]
    R, axis, pos = cut
    # Strona linii dla każdego piksela obszaru: obraz -> układ wyprostowany -> obrócony o theta.
    p = pts.reshape(-1, 2).astype(np.float32)
    q = p @ A[:, :2].T + A[:, 2]
    t = q @ R[1 - axis, :2] + R[1 - axis, 2]  # oś 0 = wiersze -> współrzędna y po obrocie
    parts = []
    for sel in (t < pos, t >= pos):
        sub = np.zeros_like(region)
        sp = p[sel].astype(np.int32)
        sub[sp[:, 1], sp[:, 0]] = 255
        sub = morph(sub, cv2.MORPH_OPEN, 3)  # odcina resztki szczeliny przy linii cięcia
        r = fit_rect(cv2.findNonZero(sub)) if cv2.countNonZero(sub) else None
        if r is None or not rect_ok(r, img_area, img_long) or \
                min(r[1]) < GAP_PART_MIN_SIDE_FRAC * img_long or \
                max(r[1]) / max(min(r[1]), 1e-6) > GAP_PART_MAX_ASPECT:
            return [rect]  # cięcie dałoby kawałek niepodobny do odbitki — rezygnujemy
        parts.append(sub)
    log.debug("Cięcie wzdłuż przerwy (głębokość %d, oś %d)", depth, axis)
    return [r for sub in parts for r in gap_cut_regions(sub, gap, gray, img_area, img_long, depth + 1)]


def find_candidates(mask: np.ndarray, gap: np.ndarray | None = None, gray: np.ndarray | None = None) -> list:
    """Zwraca listę obróconych prostokątów (w układzie kopii roboczej)."""
    h, w = mask.shape
    img_area, img_long = float(h * w), float(max(h, w))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = []
    for c in contours:
        r = fit_rect(c)
        if r[1][0] * r[1][1] < MIN_AREA_FRAC * img_area:
            continue  # kurz / śmieci
        # Zawsze sprawdzamy, czy obszar to kilka sklejonych zdjęć — także gdy jest "za duży"
        # (np. zdjęcia wypełniające prawie cały skan, rozdzielone wąskimi przerwami).
        parts = split_touching(c, img_area, img_long)
        if parts:
            log.debug("Rozdzielono zlepiony obszar na %d zdjęć", len(parts))
        elif rect_ok(r, img_area, img_long) or (GAP_SPLIT and gap is not None):
            parts = [(c, r)]
        for pc, pr in parts:
            if GAP_SPLIT and gap is not None:
                region = np.zeros_like(mask)
                cv2.drawContours(region, [pc], -1, 255, cv2.FILLED)
                rects.extend(x for x in gap_cut_regions(region, gap, gray, img_area, img_long)
                             if rect_ok(x, img_area, img_long))
            elif rect_ok(pr, img_area, img_long):
                rects.append(pr)
    return rects


# -----------------------------------------------------------------------------
# Doprecyzowanie na pełnej rozdzielczości
# -----------------------------------------------------------------------------

def refine_rect(full_bgr8: np.ndarray, rect, bg_lab: np.ndarray, others: list, edges):
    """Ponowna segmentacja w ROI wokół zgrubnego prostokąta na pełnej rozdzielczości.
    `others` to zgrubne prostokąty pozostałych zdjęć — wykluczamy je, by nie "zassać" sąsiada."""
    (cx, cy), (rw, rh), _ = rect
    H, W = full_bgr8.shape[:2]
    box = cv2.boxPoints(rect)
    margin = REFINE_MARGIN_FRAC * max(rw, rh)
    x0 = int(max(0, np.floor(box[:, 0].min() - margin)))
    y0 = int(max(0, np.floor(box[:, 1].min() - margin)))
    x1 = int(min(W, np.ceil(box[:, 0].max() + margin)))
    y1 = int(min(H, np.ceil(box[:, 1].max() + margin)))
    roi = full_bgr8[y0:y1, x0:x1]
    if roi.size == 0:
        return rect

    diff, _ = background_distance(roi, bg_lab)
    m = (diff > BG_DIFF_THRESHOLD).astype(np.uint8) * 255
    if USE_EDGES:
        m = cv2.bitwise_or(m, edge_map(roi))
    clear_scanner_edges(m, edges, offset=(x0, y0), full_shape=(H, W))
    # Ogranicz do lekko powiększonego zgrubnego prostokąta — sąsiednie zdjęcia nie przeszkadzają.
    limit = np.zeros_like(m)
    grown = ((cx - x0, cy - y0), (rw + 2 * margin, rh + 2 * margin), rect[2])
    cv2.fillPoly(limit, [np.round(cv2.boxPoints(grown)).astype(np.int32)], 255)
    for (ocx, ocy), (ow, oh), oa in others:
        other = ((ocx - x0, ocy - y0), (ow, oh), oa)
        cv2.fillPoly(limit, [np.round(cv2.boxPoints(other)).astype(np.int32)], 0)
    m = cv2.bitwise_and(m, limit)

    m = morph(m, cv2.MORPH_CLOSE, max(H, W) * CLOSE_KERNEL_FRAC)
    m = fill_holes(m)
    m = morph(m, cv2.MORPH_OPEN, max(rw, rh) * REFINE_OPEN_FRAC)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return rect
    c = max(cnts, key=cv2.contourArea)
    (ncx, ncy), (nw, nh), na = fit_rect(c)
    refined = ((ncx + x0, ncy + y0), (nw, nh), na)
    if abs(nw * nh / max(rw * rh, 1e-6) - 1.0) > REFINE_MAX_AREA_DEVIATION:
        return rect  # doprecyzowanie niewiarygodne (np. zdjęcie bardzo jasne) — zostaw zgrubne
    return refined


def grow_to_paper_edge(full_bgr8: np.ndarray, rect, bg_lab: np.ndarray):
    """
    Rozszerza prostokąt o jasny margines odbitki (np. białą ramkę zdjęcia na białym tle).

    Dla każdego boku liczymy profil: średnie "odstępstwo od tła" linii równoległych do boku,
    kolejno coraz dalej na zewnątrz. Uśrednienie po całej długości boku eliminuje szum, więc
    nawet różnica papier/pokrywa rzędu kilku ΔE jest wyraźna. Cień (ciemniejszy od tła) nie
    jest liczony jako papier. Bok rośnie, dopóki linia wyraźnie odstaje od lokalnego tła.
    """
    H, W = full_bgr8.shape[:2]
    src = order_corners(cv2.boxPoints(rect))
    ux = (src[1] - src[0]) / max(np.linalg.norm(src[1] - src[0]), 1e-6)
    uy = (src[3] - src[0]) / max(np.linalg.norm(src[3] - src[0]), 1e-6)
    rw, rh = np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[3] - src[0])
    g = int(max(4, GROW_MAX_FRAC * max(rw, rh)))
    grown = np.array([src[0] - g * ux - g * uy, src[1] + g * ux - g * uy,
                      src[2] + g * ux + g * uy, src[3] - g * ux + g * uy], np.float32)
    x0, y0 = np.maximum(np.floor(grown.min(axis=0)).astype(int) - 2, 0)
    x1, y1 = np.minimum(np.ceil(grown.max(axis=0)).astype(int) + 2, (W, H))
    roi = full_bgr8[y0:y1, x0:x1]
    if roi.size == 0:
        return rect
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32) - bg_lab
    lab[..., 0] = np.maximum(lab[..., 0], 0)  # ciemniejsze od tła (cień) nie liczy się jako papier
    paper = np.sqrt((lab ** 2).sum(axis=2))
    ow, oh = int(round(rw)) + 2 * g, int(round(rh)) + 2 * g
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
    M = cv2.getPerspectiveTransform((grown - (x0, y0)).astype(np.float32), dst)
    # Poza skanem: NaN — takie linie nie mogą zaniżać lokalnego tła ani udawać papieru.
    up = cv2.warpPerspective(paper, M, (ow, oh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
    iw, ih = ow - 2 * g, oh - 2 * g
    cx0, cx1 = g + iw // 10, g + iw - iw // 10       # środkowe 80% boku (bez narożników)
    cy0, cy1 = g + ih // 10, g + ih - ih // 10
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # linie w całości poza skanem -> NaN
        profiles = {  # od krawędzi prostokąta na zewnątrz
            "top": np.nanmean(up[g - 1::-1, cx0:cx1], axis=1),
            "bottom": np.nanmean(up[g + ih:, cx0:cx1], axis=1),
            "left": np.nanmean(up[cy0:cy1, g - 1::-1], axis=0),
            "right": np.nanmean(up[cy0:cy1, g + iw:], axis=0),
        }
    grow = {}
    for side, prof in profiles.items():
        outside = np.flatnonzero(np.isnan(prof))
        prof = prof[:outside[0]] if len(outside) else prof  # tylko linie w obrębie skanu
        if len(prof) < 4:
            grow[side] = 0
            continue
        thr = np.percentile(prof, 10) + GROW_DELTA   # lokalne tło = najniższe wartości profilu
        d = 0
        while d < len(prof) and prof[d] > thr:
            d += 1
        grow[side] = d if d < len(prof) else 0      # nie znaleziono tła -> sąsiad/krawędź, bez zmian
    if not any(grow.values()):
        return rect
    log.debug("Rozszerzenie o margines odbitki: %s", grow)
    tl = src[0] - grow["left"] * ux - grow["top"] * uy
    tr = src[1] + grow["right"] * ux - grow["top"] * uy
    br = src[2] + grow["right"] * ux + grow["bottom"] * uy
    bl = src[3] - grow["left"] * ux + grow["bottom"] * uy
    return cv2.minAreaRect(np.array([tl, tr, br, bl], np.float32))


# -----------------------------------------------------------------------------
# Prostowanie i wycinanie
# -----------------------------------------------------------------------------

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Porządkuje narożniki: lewy-górny, prawy-górny, prawy-dolny, lewy-dolny."""
    pts = pts.astype(np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def normalized_angle(rect) -> float:
    """Kąt odchylenia od siatki w zakresie (-45°, 45°]."""
    a = rect[2] % 90.0
    return a - 90.0 if a > 45.0 else a


def extract_deskewed(img: np.ndarray, rect) -> np.ndarray:
    """Wycina i prostuje zdjęcie; zachowuje typ danych (8/16 bit) i liczbę kanałów."""
    H, W = img.shape[:2]

    if abs(normalized_angle(rect)) < SKIP_ROTATION_BELOW_DEG:
        # Brak potrzeby resamplingu — czyste wycięcie, piksele identyczne z oryginałem.
        box = cv2.boxPoints(rect)
        x0 = int(max(0, round(box[:, 0].min()) + CROP_INSET_PX))
        y0 = int(max(0, round(box[:, 1].min()) + CROP_INSET_PX))
        x1 = int(min(W, round(box[:, 0].max()) - CROP_INSET_PX))
        y1 = int(min(H, round(box[:, 1].max()) - CROP_INSET_PX))
        return img[y0:y1, x0:x1].copy()

    # Narożniki źródłowe przesunięte do wewnątrz o CROP_INSET_PX (eliminuje resztki tła na brzegach).
    src = order_corners(cv2.boxPoints(rect))
    ux = (src[1] - src[0]) / max(np.linalg.norm(src[1] - src[0]), 1e-6)  # kierunek "w prawo"
    uy = (src[3] - src[0]) / max(np.linalg.norm(src[3] - src[0]), 1e-6)  # kierunek "w dół"
    i = float(CROP_INSET_PX)
    src = np.array([src[0] + i * ux + i * uy, src[1] - i * ux + i * uy,
                    src[2] - i * ux - i * uy, src[3] + i * ux - i * uy], np.float32)
    out_w = max(1, int(round(np.linalg.norm(src[1] - src[0]))))
    out_h = max(1, int(round(np.linalg.norm(src[3] - src[0]))))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    # Lanczos zachowuje ostrość; OpenCV saturuje wynik do zakresu typu (uint8/uint16).
    out = cv2.warpPerspective(img, M, (out_w, out_h), flags=INTERPOLATION,
                              borderMode=cv2.BORDER_REPLICATE)
    return out


def auto_trim(crop: np.ndarray, bg_lab: np.ndarray) -> np.ndarray:
    """Obcina z każdej strony wiersze/kolumny, które w większości wyglądają jak tło skanera."""
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return crop
    my, mx = max(1, int(h * AUTO_TRIM_MAX_FRAC)), max(1, int(w * AUTO_TRIM_MAX_FRAC))

    def bg_lines(strip: np.ndarray, axis: int) -> np.ndarray:
        # Analizujemy tylko wąskie pasy brzegowe — nie całą (np. 1200 dpi) odbitkę.
        is_bg = background_distance(to_bgr8(strip), bg_lab)[0] <= AUTO_TRIM_BG_DIFF
        return is_bg.mean(axis=axis) >= AUTO_TRIM_BG_RATIO

    def depth(flags: np.ndarray) -> int:
        nz = np.flatnonzero(~flags)
        return int(nz[0]) if len(nz) else len(flags)

    top = depth(bg_lines(crop[:my], 1))
    bottom = depth(bg_lines(crop[h - my:], 1)[::-1])
    left = depth(bg_lines(crop[:, :mx], 0))
    right = depth(bg_lines(crop[:, w - mx:], 0)[::-1])
    return crop[top:h - bottom, left:w - right]


def reading_order(dets: list[Detection]) -> list[Detection]:
    """Sortuje zdjęcia rzędami (góra->dół), a w rzędzie od lewej do prawej."""
    dets = sorted(dets, key=lambda d: d.center[1])
    rows: list[list[Detection]] = []
    for d in dets:
        if rows:
            ref = rows[-1][0]
            if abs(d.center[1] - ref.center[1]) < ROW_GROUPING_FRAC * min(ref.size):
                rows[-1].append(d)
                continue
        rows.append([d])
    return [d for row in rows for d in sorted(row, key=lambda d: d.center[0])]


# -----------------------------------------------------------------------------
# Orientacja wg twarzy
# -----------------------------------------------------------------------------

_face_detector = None  # leniwie tworzony w każdym procesie roboczym; False = niedostępny


def get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = False
        if not FACE_MODEL_PATH.is_file():
            log.warning("Brak modelu twarzy %s — automatyczna orientacja wyłączona", FACE_MODEL_PATH)
            return None
        try:
            try:  # wycisza ostrzeżenia OpenCV DNN (zostawia tylko błędy)
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            except AttributeError:
                pass
            _face_detector = cv2.FaceDetectorYN.create(str(FACE_MODEL_PATH), "", (320, 320),
                                                       FACE_SCORE_THRESHOLD, 0.3, 500)
        except (cv2.error, AttributeError) as exc:
            log.warning("Nie można załadować modelu twarzy (%s) — orientacja wyłączona", exc)
    return _face_detector or None


ROTATION_NAMES = {0: "bez obrotu", 1: "90 w lewo", 2: "180", 3: "90 w prawo"}  # k jak w np.rot90


def face_score(det, img8: np.ndarray):
    """(suma pewności, najwyższa pewność, twarze [x,y,w,h,pewność]) dla twarzy stojących pionowo."""
    det.setInputSize((img8.shape[1], img8.shape[0]))
    _, faces = det.detect(img8)
    total = best = 0.0
    found = []
    for f in faces if faces is not None else []:
        # YuNet: [x,y,w,h, oko P, oko L, nos, usta P, usta L (x,y), pewność]
        eyes_y, nose_y, mouth_y = (f[5] + f[7]) / 2, f[9], (f[11] + f[13]) / 2
        if not eyes_y < nose_y < mouth_y:
            continue  # układ punktów nie pasuje do pionowej twarzy
        total += float(f[14])
        best = max(best, float(f[14]))
        found.append([float(v) for v in f[:4]] + [float(f[14])])
    return total, best, found


def enhance_for_faces(img8: np.ndarray) -> np.ndarray:
    """Wyrównanie kontrastu jasności (CLAHE) — pomaga na ciemnych, wyblakłych odbitkach."""
    lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def face_orientation(crop: np.ndarray) -> tuple[int, dict]:
    """
    Liczba obrotów o 90° (przeciwnie do wskazówek zegara, jak np.rot90), po których twarze
    stoją pionowo; 0 = bez zmian (także gdy na zdjęciu nie ma pewnie wykrytych twarzy).
    Drugi element to opis decyzji (do podglądu --debug).
    """
    info = {"scores": [0.0] * 4, "side": None, "faces": [], "reason": "brak twarzy"}
    det = get_face_detector()
    if det is None:
        info["reason"] = "detektor niedostepny"
        return 0, info
    img8 = to_bgr8(crop)
    for side in FACE_DETECT_SIDES:
        f = min(1.0, side / max(img8.shape[:2]))
        small = cv2.resize(img8, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img8
        if FACE_ENHANCE:
            small = enhance_for_faces(small)
        res = [face_score(det, np.ascontiguousarray(np.rot90(small, k))) for k in range(4)]
        # Decyduje najpewniejsza pojedyncza twarz (suma tylko rozstrzyga remisy) — fałszywe
        # trafienia w złej orientacji mają zwykle niską pewność, ale bywa ich kilka.
        k = max(range(4), key=lambda i: (res[i][1], res[i][0]))
        best = res[k][1]
        info["scores"] = [r[1] for r in res]
        info["side"] = side
        if best < FACE_MIN_CONFIDENCE:
            if f >= 1:
                break  # zdjęcie i tak mniejsze — większa skala nic nie da
            continue  # brak pewnej twarzy — spróbuj w większej rozdzielczości
        if k and best < res[0][1] + FACE_ROTATION_MARGIN:
            info["reason"] = "za mala przewaga nad obecna orientacja"
            k = 0
        else:
            info["reason"] = "twarz %.2f" % best
        # Twarze we współrzędnych względnych (0..1) obrazu po obrocie o k.
        rh, rw = np.rot90(small, k).shape[:2]
        info["faces"] = [[x / rw, y / rh, w / rw, h / rh, sc] for x, y, w, h, sc in res[k][2]]
        return k, info
    return 0, info


# -----------------------------------------------------------------------------
# Przetwarzanie jednego skanu
# -----------------------------------------------------------------------------

def scale_rect(rect, f: float):
    (cx, cy), (w, h), a = rect
    return (cx * f, cy * f), (w * f, h * f), a


def detect(img: np.ndarray):
    """
    Zwraca (detekcje w układzie pełnej rozdzielczości, maska robocza, kolor tła,
    kopia "mid" 8-bit do podglądu, skala mid względem oryginału).
    """
    full8 = to_bgr8(img)
    H, W = full8.shape[:2]
    # Kopia "mid" (do REFINE_MAX_SIDE px) — doprecyzowanie bez trzymania kilku kopii 400 MB w RAM.
    s_mid = min(1.0, REFINE_MAX_SIDE / max(H, W))
    mid = cv2.resize(full8, (round(W * s_mid), round(H * s_mid)), interpolation=cv2.INTER_AREA) \
        if s_mid < 1.0 else full8
    del full8
    scale = min(1.0, WORK_MAX_SIDE / max(H, W))  # robocza względem oryginału
    work = cv2.resize(mid, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA) \
        if scale < s_mid else mid

    mask, bg_lab, edges, gap = build_foreground_mask(work)
    w2m = s_mid / scale  # robocza -> mid
    edges_mid = tuple(int(np.ceil(e * w2m)) for e in edges)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    coarse = [scale_rect(r, w2m) for r in find_candidates(mask, gap, gray)]
    dets = []
    for k, rect in enumerate(coarse):
        if REFINE_FULL_RES and w2m > 1.0:
            rect = refine_rect(mid, rect, bg_lab, coarse[:k] + coarse[k + 1:], edges_mid)
        if GROW_TO_PAPER_EDGE:
            rect = grow_to_paper_edge(mid, rect, bg_lab)
        dets.append(Detection(*scale_rect(rect, 1.0 / s_mid)))
    return reading_order(dets), mask, bg_lab, mid, s_mid


def save_debug(mid: np.ndarray, s_mid: float, dets: list[Detection], mask: np.ndarray,
               notes: list[str], path: Path) -> None:
    """Podgląd detekcji: ramki z numerami i informacją o obrocie + maska robocza."""
    scale = min(1.0, 2000 / max(mid.shape[:2]))
    vis = cv2.resize(mid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    f = s_mid * scale
    thick = max(2, max(vis.shape[:2]) // 500)
    for n, d in enumerate(dets, 1):
        box = np.round(cv2.boxPoints(scale_rect(d.rect, f))).astype(np.int32)
        cv2.polylines(vis, [box], True, (0, 0, 255), thick)
        x0, y0 = box.min(axis=0)
        label = f"{n}" + (f": {notes[n - 1]}" if n - 1 < len(notes) and notes[n - 1] else "")
        org = (int(x0) + 8 * thick, int(y0) + 22 * thick)  # lewy górny róg ramki, wewnątrz
        for color, t in (((255, 255, 255), thick * 4), ((0, 0, 200), thick * 2)):
            cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * thick, color, t)
    m = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (vis.shape[1], vis.shape[0]))
    ok, buf = cv2.imencode(".jpg", np.hstack([vis, m]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        buf.tofile(str(path))


def face_thumb(crop: np.ndarray, k: int, info: dict, n: int, height: int = 360) -> np.ndarray:
    """Miniatura wycinka (już obróconego) z ramkami twarzy i opisem decyzji o obrocie."""
    img = to_bgr8(crop)
    f = height / img.shape[0]
    t = cv2.resize(img, (max(1, int(img.shape[1] * f)), height), interpolation=cv2.INTER_AREA)
    t = cv2.copyMakeBorder(t, 0, 70, 0, max(0, 330 - t.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))
    th, tw = height, int(img.shape[1] * f)
    for x, y, w, h, sc in info.get("faces", []):
        p0, p1 = (int(x * tw), int(y * th)), (int((x + w) * tw), int((y + h) * th))
        cv2.rectangle(t, p0, p1, (0, 220, 0), 2)
        cv2.putText(t, "%.2f" % sc, (p0[0], max(12, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    sc = info.get("scores", [0] * 4)
    lines = [f"#{n}: {ROTATION_NAMES[k]}  ({info.get('reason', '')})",
             "0:%.2f  L90:%.2f  180:%.2f  P90:%.2f" % tuple(sc)]
    for i, line in enumerate(lines):
        cv2.putText(t, line, (6, height + 25 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return t


def save_face_sheet(thumbs: list[np.ndarray], path: Path, per_row: int = 4) -> None:
    if not thumbs:
        return
    rows = []
    for i in range(0, len(thumbs), per_row):
        row = [cv2.copyMakeBorder(t, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(90, 90, 90)) for t in thumbs[i:i + per_row]]
        rows.append(np.hstack(row))
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(90, 90, 90)) for r in rows]
    ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        buf.tofile(str(path))


def process_scan(path: Path, out_dir: Path, fmt: str, debug: bool,
                 rotate_faces: bool = AUTO_ROTATE_FACES) -> tuple[str, int, int, str | None]:
    """Przetwarza jeden plik. Zwraca (nazwa, liczba_zdjęć, liczba_obróconych, błąd)."""
    cv2.setNumThreads(1)  # równoległość zapewnia pula procesów
    try:
        img = read_image(path)
        if img is None:
            return path.name, 0, 0, "nie można odczytać pliku"
        dets, mask, bg_lab, mid, s_mid = detect(img)
        ext = "png" if fmt == "png" else "tif"
        saved = rotated = 0
        notes, thumbs = [], []
        for d in dets:
            crop = extract_deskewed(img, d.rect)
            if AUTO_TRIM_BORDERS:
                crop = auto_trim(crop, bg_lab)
            if crop.size == 0:
                notes.append("puste")
                continue
            k, info = face_orientation(crop) if rotate_faces else (0, {})
            if k:
                crop = np.ascontiguousarray(np.rot90(crop, k))  # obrót o 90° — bezstratny
                rotated += 1
            saved += 1
            name = CROP_NAME_PATTERN.format(stem=path.stem, index=saved, ext=ext)
            write_image(out_dir / name, crop, fmt)
            log.debug("%s: %s (%s)", name, ROTATION_NAMES[k], info.get("reason", "-"))
            notes.append(f"obrot {ROTATION_NAMES[k]}" if k else "")
            if debug and rotate_faces:
                thumbs.append(face_thumb(crop, k, info, saved))
        if debug:
            save_debug(mid, s_mid, dets, mask, notes, out_dir / f"{path.stem}_debug.jpg")
            save_face_sheet(thumbs, out_dir / f"{path.stem}_debug_twarze.jpg")
        return path.name, saved, rotated, None
    except Exception as exc:  # pojedynczy uszkodzony plik nie przerywa wsadu
        log.debug("Błąd przy %s", path, exc_info=True)
        return path.name, 0, 0, f"{type(exc).__name__}: {exc}"


# -----------------------------------------------------------------------------
# CLI / batch
# -----------------------------------------------------------------------------

def collect_inputs(in_dir: Path, recursive: bool) -> list[Path]:
    it = in_dir.rglob("*") if recursive else in_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Automatyczne dzielenie skanów na pojedyncze, wyprostowane zdjęcia.")
    ap.add_argument("input", type=Path, help="folder ze skanami")
    ap.add_argument("output", type=Path, help="folder wynikowy")
    ap.add_argument("--format", choices=["png", "tiff"], default=OUTPUT_FORMAT, help="format bezstratny")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="liczba procesów")
    ap.add_argument("--recursive", action="store_true", help="przeszukuj podfoldery")
    ap.add_argument("--debug", action="store_true", help="zapisz podgląd detekcji (*_debug.jpg)")
    ap.add_argument("--no-face-rotate", dest="rotate_faces", action="store_false",
                    default=AUTO_ROTATE_FACES, help="nie obracaj zdjęć na podstawie twarzy")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if not args.input.is_dir():
        log.error("Folder wejściowy nie istnieje: %s", args.input)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    files = collect_inputs(args.input, args.recursive)
    if not files:
        log.warning("Brak obsługiwanych plików w %s", args.input)
        return 0

    if args.rotate_faces and not FACE_MODEL_PATH.is_file():
        log.warning("Brak modelu twarzy %s — automatyczna orientacja wyłączona", FACE_MODEL_PATH)
        args.rotate_faces = False
    log.info("Skanów: %d | format: %s | procesy: %d | orientacja wg twarzy: %s",
             len(files), args.format, args.workers, "tak" if args.rotate_faces else "nie")
    t0, total, total_rot, failed = time.perf_counter(), 0, 0, 0

    def report(name, n, rot, err):
        nonlocal total, total_rot, failed
        if err:
            failed += 1
            log.error("%-40s BŁĄD: %s", name, err)
        else:
            total += n
            total_rot += rot
            extra = f" (obrócone wg twarzy: {rot})" if rot else ""
            (log.warning if n == 0 else log.info)("%-40s zdjęć: %d%s", name, n, extra)

    job = (args.output, args.format, args.debug, args.rotate_faces)
    if args.workers <= 1:
        for f in files:
            report(*process_scan(f, *job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_scan, f, *job) for f in files]
            for fut in as_completed(futures):
                report(*fut.result())

    log.info("Gotowe: %d zdjęć z %d skanów w %.1fs (obrócone: %d, błędy: %d)",
             total, len(files), time.perf_counter() - t0, total_rot, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
