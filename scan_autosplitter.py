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
         - filtracja konturów: pole, wymiary, prostokątność.
    3. Doprecyzowanie każdego prostokąta na pełnej rozdzielczości (minAreaRect otoczki wypukłej)
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
WORK_MAX_SIDE = 1600             # dłuższy bok kopii roboczej [px]; większy = dokładniej, wolniej
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

# --- Doprecyzowanie i prostowanie --------------------------------------------
REFINE_FULL_RES = True           # doprecyzuj krawędzie na pełnej rozdzielczości
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
FACE_DETECT_SIDES = (640, 1024)  # rozmiary (dłuższy bok), na których szukamy twarzy — kolejno; większy
                                 # tylko gdy na mniejszym brak pewnej twarzy (małe twarze, zdjęcia grupowe)
FACE_SCORE_THRESHOLD = 0.6       # min. pewność pojedynczej twarzy branej pod uwagę
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
    return mask, bg_lab, edges


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


def find_candidates(mask: np.ndarray) -> list:
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
            rects.extend(r for _, r in parts)
        elif rect_ok(r, img_area, img_long):
            if rectangularity(c, r) < MIN_RECTANGULARITY:
                log.debug("Kształt nieprostokątny (%.2f) — używam minAreaRect otoczki", rectangularity(c, r))
            rects.append(r)
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
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return crop
    is_bg = background_distance(to_bgr8(crop), bg_lab)[0] <= AUTO_TRIM_BG_DIFF
    h, w = is_bg.shape
    max_y, max_x = int(h * AUTO_TRIM_MAX_FRAC), int(w * AUTO_TRIM_MAX_FRAC)
    top = bottom = left = right = 0
    while top < max_y and is_bg[top].mean() >= AUTO_TRIM_BG_RATIO:
        top += 1
    while bottom < max_y and is_bg[h - 1 - bottom].mean() >= AUTO_TRIM_BG_RATIO:
        bottom += 1
    while left < max_x and is_bg[:, left].mean() >= AUTO_TRIM_BG_RATIO:
        left += 1
    while right < max_x and is_bg[:, w - 1 - right].mean() >= AUTO_TRIM_BG_RATIO:
        right += 1
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


def face_score(det, img8: np.ndarray) -> tuple[float, float]:
    """(suma pewności, najwyższa pewność) twarzy ustawionych pionowo w danym obrazie."""
    det.setInputSize((img8.shape[1], img8.shape[0]))
    _, faces = det.detect(img8)
    total = best = 0.0
    for f in faces if faces is not None else []:
        # YuNet: [x,y,w,h, oko P, oko L, nos, usta P, usta L (x,y), pewność]
        eyes_y, nose_y, mouth_y = (f[5] + f[7]) / 2, f[9], (f[11] + f[13]) / 2
        if not eyes_y < nose_y < mouth_y:
            continue  # układ punktów nie pasuje do pionowej twarzy
        total += float(f[14])
        best = max(best, float(f[14]))
    return total, best


def face_orientation(crop: np.ndarray) -> int:
    """Liczba obrotów o 90° (przeciwnie do wskazówek zegara, jak np.rot90), po których twarze
    stoją pionowo. 0 = bez zmian (także gdy na zdjęciu nie ma pewnie wykrytych twarzy)."""
    det = get_face_detector()
    if det is None:
        return 0
    img8 = to_bgr8(crop)
    for side in FACE_DETECT_SIDES:
        f = min(1.0, side / max(img8.shape[:2]))
        small = cv2.resize(img8, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img8
        scores = [face_score(det, np.ascontiguousarray(np.rot90(small, k))) for k in range(4)]
        # Decyduje najpewniejsza pojedyncza twarz (suma tylko rozstrzyga remisy) — fałszywe
        # trafienia w złej orientacji mają zwykle niską pewność, ale bywa ich kilka.
        k = max(range(4), key=lambda i: (scores[i][1], scores[i][0]))
        best = scores[k][1]
        if best < FACE_MIN_CONFIDENCE:
            continue  # brak pewnej twarzy — spróbuj w większej rozdzielczości
        return k if k and best >= scores[0][1] + FACE_ROTATION_MARGIN else 0
    return 0


# -----------------------------------------------------------------------------
# Przetwarzanie jednego skanu
# -----------------------------------------------------------------------------

def detect(img: np.ndarray) -> tuple[list[Detection], np.ndarray, np.ndarray]:
    full8 = to_bgr8(img)
    H, W = full8.shape[:2]
    scale = min(1.0, WORK_MAX_SIDE / max(H, W))
    work = cv2.resize(full8, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else full8

    mask, bg_lab, edges = build_foreground_mask(work)
    edges_full = tuple(int(np.ceil(e / scale)) for e in edges)
    coarse = [((cx / scale, cy / scale), (rw / scale, rh / scale), ang)
              for (cx, cy), (rw, rh), ang in find_candidates(mask)]
    dets = []
    for k, rect in enumerate(coarse):
        if REFINE_FULL_RES and scale < 1.0:
            rect = refine_rect(full8, rect, bg_lab, coarse[:k] + coarse[k + 1:], edges_full)
        if GROW_TO_PAPER_EDGE:
            rect = grow_to_paper_edge(full8, rect, bg_lab)
        dets.append(Detection(*rect))
    return reading_order(dets), mask, bg_lab


def save_debug(img: np.ndarray, dets: list[Detection], mask: np.ndarray, path: Path) -> None:
    vis = to_bgr8(img).copy()
    thick = max(2, max(vis.shape[:2]) // 400)
    for n, d in enumerate(dets, 1):
        box = np.round(cv2.boxPoints(d.rect)).astype(np.int32)
        cv2.polylines(vis, [box], True, (0, 0, 255), thick)
        cv2.putText(vis, str(n), (int(d.center[0]), int(d.center[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    thick, (0, 0, 255), thick * 2)
    scale = min(1.0, 2000 / max(vis.shape[:2]))
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (vis.shape[1], vis.shape[0]))
    ok, buf = cv2.imencode(".jpg", np.hstack([vis, m]), [cv2.IMWRITE_JPEG_QUALITY, 85])
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
        dets, mask, bg_lab = detect(img)
        ext = "png" if fmt == "png" else "tif"
        saved = rotated = 0
        for d in dets:
            crop = extract_deskewed(img, d.rect)
            if AUTO_TRIM_BORDERS:
                crop = auto_trim(crop, bg_lab)
            if crop.size == 0:
                continue
            if rotate_faces:
                k = face_orientation(crop)
                if k:
                    crop = np.ascontiguousarray(np.rot90(crop, k))  # obrót o 90° — bezstratny
                    rotated += 1
            saved += 1
            write_image(out_dir / CROP_NAME_PATTERN.format(stem=path.stem, index=saved, ext=ext), crop, fmt)
        if debug:
            save_debug(img, dets, mask, out_dir / f"{path.stem}_debug.jpg")
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
