#!/usr/bin/env python3
"""
scan_autosplitter.py — automatyczne wykrywanie, wycinanie i prostowanie zdjęć ze skanów.

Pipeline (klasyczne CV, OpenCV + NumPy):
    1. Wczytanie skanu z zachowaniem głębi bitowej (8/16 bit, TIFF/PNG/JPG).
    2. Detekcja na pomniejszonej kopii roboczej:
         - model tła skanera (mediana koloru w pasie brzegowym, przestrzeń Lab),
         - maska "odległości od tła" (ΔE w Lab) + krawędzie Canny,
         - domknięcie morfologiczne, wypełnienie dziur, otwarcie (usuwa kurz i linie),
         - filtracja konturów: pole, wymiary, prostokątność,
         - rozdzielanie stykających się zdjęć (transformata odległości + watershed).
    3. Doprecyzowanie każdego prostokąta na pełnej rozdzielczości (minAreaRect otoczki wypukłej).
    4. Deskew: transformacja perspektywiczna prostokąta obróconego -> prostokąt osiowy
       (Lanczos), opcjonalne przycięcie resztek tła na brzegach. Kąt ~0° = czyste wycięcie,
       piksele identyczne bit w bit z oryginałem.
    5. Zapis bezstratny (PNG / nieskompresowany TIFF) jako <skan>_crop_NN.<ext>.

Użycie:
    python scan_autosplitter.py WEJSCIE WYJSCIE [--format png|tiff] [--workers N] [--debug]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
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
WORK_MAX_SIDE = 1600             # dłuższy bok kopii roboczej [px]; większy = dokładniej, wolniej
BG_SAMPLE_BORDER_FRAC = 0.03     # szerokość pasa brzegowego do estymacji koloru tła (ułamek boku)
BG_DIFF_THRESHOLD = 14.0         # min. odległość koloru od tła (ΔE w Lab, skala 0-255) = "zdjęcie"
BG_DIFF_BLUR = 5                 # rozmycie mapy różnic (tłumi szum/ziarno skanera), nieparzyste
CANNY_LOW = 30                   # dolny próg Canny (krawędzie jasnych zdjęć na jasnym tle)
CANNY_HIGH = 90                  # górny próg Canny
USE_EDGES = True                 # łącz maskę koloru z krawędziami (pomaga przy jasnych zdjęciach)
CLOSE_KERNEL_FRAC = 0.012        # rozmiar jądra domknięcia (ułamek dłuższego boku roboczego)
OPEN_KERNEL_FRAC = 0.006         # rozmiar jądra otwarcia — usuwa kurz, włosy, linie krawędzi skanera
SCANNER_EDGE_IGNORE_FRAC = 0.003 # pas przy krawędziach skanu ignorowany w detekcji (ciemna ramka/cień
                                 # pokrywy skanera "przyklejałby się" do zdjęć dosuniętych do rogu)

# --- Filtracja kandydatów ----------------------------------------------------
MIN_AREA_FRAC = 0.008            # min. pole zdjęcia jako ułamek pola skanu (odrzuca śmieci)
MAX_AREA_FRAC = 0.98             # max. pole (odrzuca "cały skan" jako jedno zdjęcie)
MIN_SIDE_FRAC = 0.04             # min. długość krótszego boku jako ułamek dłuższego boku skanu
MAX_ASPECT_RATIO = 8.0           # max. proporcje boków (odrzuca paski/listwy skanera)
MIN_RECTANGULARITY = 0.85        # min. pole konturu / pole minAreaRect dla części po rozdzieleniu
SPLIT_CORE_LEVELS = (0.5, 0.6, 0.7, 0.8)  # progi transformaty odległości (ułamek maks.) do szukania "jąder" zdjęć
SPLIT_MIN_PART_FRAC = 0.12       # część po rozdzieleniu musi mieć >= tyle pola całego obszaru

# --- Doprecyzowanie i prostowanie --------------------------------------------
REFINE_FULL_RES = True           # doprecyzuj krawędzie na pełnej rozdzielczości
REFINE_MARGIN_FRAC = 0.03        # margines ROI przy doprecyzowaniu (ułamek boku zdjęcia)
REFINE_OPEN_FRAC = 0.02          # otwarcie przy doprecyzowaniu (ułamek boku zdjęcia) — odcina kurz
                                 # przyklejony do krawędzi, który zawyżałby otoczkę wypukłą
REFINE_MAX_AREA_DEVIATION = 0.12 # gdy doprecyzowany prostokąt różni się bardziej -> użyj zgrubnego
CROP_INSET_PX = 2                # stałe przycięcie do wewnątrz [px pełnej rozdz.] (anty-ramka)
AUTO_TRIM_BORDERS = True         # dodatkowo przytnij wiersze/kolumny brzegowe wyglądające jak tło
AUTO_TRIM_MAX_FRAC = 0.005       # max. przycięcie automatyczne z każdej strony (ułamek boku); mało,
                                 # bo biała ramka odbitki wygląda jak tło i nie może zostać ścięta
AUTO_TRIM_BG_RATIO = 0.6         # linia jest "tłem", gdy >= tyle jej pikseli jest podobnych do tła
SKIP_ROTATION_BELOW_DEG = 0.05   # poniżej tego kąta tylko wycinamy (bit-w-bit identyczne piksele)
INTERPOLATION = cv2.INTER_LANCZOS4

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
    """Kolor tła = mediana pikseli z pasa brzegowego (odporna na kurz i krawędź skanera)."""
    h, w = lab.shape[:2]
    b = max(2, int(min(h, w) * BG_SAMPLE_BORDER_FRAC))
    strip = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ])
    # Tło skanera jest jasne — ignorujemy ciemne piksele (zdjęcia przy krawędzi, cień pokrywy).
    bright = strip[strip[:, 0] >= np.percentile(strip[:, 0], 40)]
    return np.median(bright if len(bright) else strip, axis=0).astype(np.float32)


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


def clear_scanner_edges(mask: np.ndarray, full_long_side: float, scale: float = 1.0,
                        offset: tuple[int, int] = (0, 0), full_shape: tuple[int, int] | None = None) -> None:
    """Zeruje (in-place) pas SCANNER_EDGE_IGNORE_FRAC przy krawędziach całego skanu.
    Dla ROI podajemy jego przesunięcie i wymiary pełnego skanu."""
    e = int(np.ceil(SCANNER_EDGE_IGNORE_FRAC * full_long_side * scale))
    if e <= 0:
        return
    h, w = mask.shape
    H, W = full_shape or (h, w)
    x0, y0 = offset
    mask[:, :max(0, e - x0)] = 0
    mask[:max(0, e - y0), :] = 0
    mask[:, max(0, W - e - x0):] = 0
    mask[max(0, H - e - y0):, :] = 0


def build_foreground_mask(bgr8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binarna maska obszarów będących zdjęciami (na kopii roboczej)."""
    diff, bg_lab = background_distance(bgr8)
    mask = (diff > BG_DIFF_THRESHOLD).astype(np.uint8) * 255
    if USE_EDGES:
        mask = cv2.bitwise_or(mask, edge_map(bgr8))
    clear_scanner_edges(mask, max(mask.shape))

    long_side = max(mask.shape)
    kc = odd(long_side * CLOSE_KERNEL_FRAC)
    ko = odd(long_side * OPEN_KERNEL_FRAC)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kc, kc)))
    mask = fill_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ko, ko)))
    return mask, bg_lab


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


def split_touching(component: np.ndarray, img_area: float, img_long: float) -> list:
    """
    Rozdziela stykające się / sklejone zdjęcia (transformata odległości + watershed).

    W transformacie odległości każda odbitka tworzy własne "jądro" (maksimum w środku),
    a miejsce styku to wąskie gardło o niższej wartości. Progujemy mapę na kilku poziomach;
    jeśli powstaje >=2 jąder, watershed rozrasta je z powrotem do granic maski, tnąc obszar
    dokładnie w przewężeniu. Pojedyncza wypukła odbitka ma jedno jądro, więc nie jest dzielona.
    """
    dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    comp_area = float(cv2.countNonZero(component))
    for level in SPLIT_CORE_LEVELS:
        n, cores = cv2.connectedComponents((dist > level * dist.max()).astype(np.uint8))
        if n - 1 < 2:
            continue
        markers = flood_from_cores(cores, dist)
        parts = []
        for lbl in range(1, n):
            part = (markers == lbl).astype(np.uint8) * 255
            if cv2.countNonZero(part) < SPLIT_MIN_PART_FRAC * comp_area:
                break
            cnts, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cnts, key=cv2.contourArea)
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
        if not rect_ok(r, img_area, img_long):
            continue
        # Zdjęcia leżące blisko siebie bywają sklejone domknięciem morfologicznym
        # (wąski "mostek") — zawsze sprawdzamy, czy obszar daje się rozdzielić.
        comp = np.zeros_like(mask)
        cv2.drawContours(comp, [c], -1, 255, cv2.FILLED)
        parts = split_touching(comp, img_area, img_long)
        if parts:
            log.debug("Rozdzielono zlepiony obszar na %d zdjęć", len(parts))
            rects.extend(r for _, r in parts)
            continue
        if rectangularity(c, r) < MIN_RECTANGULARITY:
            log.debug("Kształt nieprostokątny (%.2f) — używam minAreaRect otoczki", rectangularity(c, r))
        rects.append(r)
    return rects


# -----------------------------------------------------------------------------
# Doprecyzowanie na pełnej rozdzielczości
# -----------------------------------------------------------------------------

def refine_rect(full_bgr8: np.ndarray, rect, bg_lab: np.ndarray, others: list):
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
    clear_scanner_edges(m, max(H, W), offset=(x0, y0), full_shape=(H, W))
    # Ogranicz do lekko powiększonego zgrubnego prostokąta — sąsiednie zdjęcia nie przeszkadzają.
    limit = np.zeros_like(m)
    grown = ((cx - x0, cy - y0), (rw + 2 * margin, rh + 2 * margin), rect[2])
    cv2.fillPoly(limit, [np.round(cv2.boxPoints(grown)).astype(np.int32)], 255)
    for (ocx, ocy), (ow, oh), oa in others:
        shrunk = ((ocx - x0, ocy - y0), (ow * 1.0, oh * 1.0), oa)
        cv2.fillPoly(limit, [np.round(cv2.boxPoints(shrunk)).astype(np.int32)], 0)
    m = cv2.bitwise_and(m, limit)

    k = odd(max(rw, rh) * CLOSE_KERNEL_FRAC)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    m = fill_holes(m)
    ko = odd(max(rw, rh) * REFINE_OPEN_FRAC)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ko, ko)))

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return rect
    c = max(cnts, key=cv2.contourArea)
    (ncx, ncy), (nw, nh), na = fit_rect(c)
    refined = ((ncx + x0, ncy + y0), (nw, nh), na)
    if abs(nw * nh / max(rw * rh, 1e-6) - 1.0) > REFINE_MAX_AREA_DEVIATION:
        return rect  # doprecyzowanie niewiarygodne (np. zdjęcie bardzo jasne) — zostaw zgrubne
    return refined


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
    is_bg = background_distance(to_bgr8(crop), bg_lab)[0] <= BG_DIFF_THRESHOLD
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
# Przetwarzanie jednego skanu
# -----------------------------------------------------------------------------

def detect(img: np.ndarray) -> tuple[list[Detection], np.ndarray, np.ndarray]:
    full8 = to_bgr8(img)
    H, W = full8.shape[:2]
    scale = min(1.0, WORK_MAX_SIDE / max(H, W))
    work = cv2.resize(full8, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else full8

    mask, bg_lab = build_foreground_mask(work)
    coarse = [((cx / scale, cy / scale), (rw / scale, rh / scale), ang)
              for (cx, cy), (rw, rh), ang in find_candidates(mask)]
    dets = []
    for k, rect in enumerate(coarse):
        if REFINE_FULL_RES and scale < 1.0:
            rect = refine_rect(full8, rect, bg_lab, coarse[:k] + coarse[k + 1:])
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


def process_scan(path: Path, out_dir: Path, fmt: str, debug: bool) -> tuple[str, int, str | None]:
    """Przetwarza jeden plik. Zwraca (nazwa, liczba_zdjęć, błąd)."""
    cv2.setNumThreads(1)  # równoległość zapewnia pula procesów
    try:
        img = read_image(path)
        if img is None:
            return path.name, 0, "nie można odczytać pliku"
        dets, mask, bg_lab = detect(img)
        ext = "png" if fmt == "png" else "tif"
        saved = 0
        for d in dets:
            crop = extract_deskewed(img, d.rect)
            if AUTO_TRIM_BORDERS:
                crop = auto_trim(crop, bg_lab)
            if crop.size == 0:
                continue
            saved += 1
            write_image(out_dir / CROP_NAME_PATTERN.format(stem=path.stem, index=saved, ext=ext), crop, fmt)
        if debug:
            save_debug(img, dets, mask, out_dir / f"{path.stem}_debug.jpg")
        return path.name, saved, None
    except Exception as exc:  # pojedynczy uszkodzony plik nie przerywa wsadu
        log.debug("Błąd przy %s", path, exc_info=True)
        return path.name, 0, f"{type(exc).__name__}: {exc}"


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

    log.info("Skanów: %d | format: %s | procesy: %d", len(files), args.format, args.workers)
    t0, total, failed = time.perf_counter(), 0, 0

    def report(name, n, err):
        nonlocal total, failed
        if err:
            failed += 1
            log.error("%-40s BŁĄD: %s", name, err)
        else:
            total += n
            (log.warning if n == 0 else log.info)("%-40s zdjęć: %d", name, n)

    if args.workers <= 1:
        for f in files:
            report(*process_scan(f, args.output, args.format, args.debug))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_scan, f, args.output, args.format, args.debug) for f in files]
            for fut in as_completed(futures):
                report(*fut.result())

    log.info("Gotowe: %d zdjęć z %d skanów w %.1fs (błędy: %d)",
             total, len(files), time.perf_counter() - t0, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
