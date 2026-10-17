# ScanAutoSplitter

Automatyczne dzielenie skanów na pojedyncze zdjęcia. Skrypt bierze folder ze skanami
(na każdym skanie kilka odbitek, często ułożonych krzywo), wykrywa każde zdjęcie,
wycina je, prostuje i zapisuje jako osobny plik w formacie bezstratnym (PNG lub TIFF).

```
skany/                         wyniki/
  album_001.tif   ──────▶       album_001_crop_01.png
                                album_001_crop_02.png
                                album_001_crop_03.png
```

- Obsługiwane formaty wejściowe: **TIF/TIFF, PNG, JPG/JPEG, BMP** (8 i 16 bit, kolor i skala szarości).
- Wyjście: **PNG** (domyślnie) lub **nieskompresowany TIFF**, z zachowaniem głębi bitowej.
- Radzi sobie ze zdjęciami dosuniętymi do krawędzi i rogów skanu, leżącymi blisko siebie
  (od ok. 2 mm przerwy) i z białą ramką odbitki na białym tle skanera.
- **Automatyczna orientacja wg twarzy:** zdjęcie leżące bokiem lub do góry nogami jest
  obracane o 90/180/270°, tak aby twarze stały pionowo (obrót o 90° jest bezstratny).
- Działa w całości **lokalnie** na zwykłym CPU, bez GPU i bez internetu (OpenCV + NumPy;
  detektor twarzy YuNet — plik `models/face_detection_yunet_2023mar.onnx`, 230 KB, licencja MIT,
  dołączony do repozytorium).

---

## 1. Wymagania

- **Python 3.9 lub nowszy** — sprawdź wersję:
  ```bash
  python --version        # Windows
  python3 --version       # Linux / macOS
  ```
  Jeśli Pythona nie ma: pobierz go z <https://www.python.org/downloads/>.
  **Windows:** w instalatorze zaznacz **„Add python.exe to PATH”**.
- **Git** (opcjonalnie, tylko do sklonowania repozytorium): <https://git-scm.com/downloads>.

> Na Windows używaj polecenia `python`, na Linux/macOS zwykle `python3`.
> W przykładach poniżej oba warianty są podane tam, gdzie się różnią.

---

## 2. Pobranie repozytorium

### Opcja A — przez Git

```bash
git clone https://github.com/chudini122/ScanAutoSplitter.git
cd ScanAutoSplitter
```

Aktualizacja do najnowszej wersji w przyszłości:

```bash
git pull
```

### Opcja B — bez Gita

Na stronie <https://github.com/chudini122/ScanAutoSplitter> kliknij
**Code → Download ZIP**, rozpakuj archiwum i otwórz terminal w rozpakowanym folderze.

---

## 3. Pierwsze uruchomienie — instalacja bibliotek

Wybierz **jeden** z dwóch wariantów. Zalecany jest wariant ze środowiskiem wirtualnym —
biblioteki instalują się wtedy tylko dla tego projektu i nie mieszają się z resztą systemu.

### Wariant 1 (zalecany): środowisko wirtualne `venv`

> **Ważne:** zawsze instaluj przez `python -m pip …`, a nie samo `pip …`. Gdy na komputerze
> jest kilka Pythonów, `python` i `pip` mogą wskazywać na **różne** instalacje — biblioteki
> trafią wtedy do innego Pythona niż ten, który uruchamia skrypt (patrz sekcja 6).

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Jeśli PowerShell zgłosi błąd „running scripts is disabled on this system”, wykonaj raz:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
> i spróbuj ponownie aktywować środowisko.

**Windows (cmd.exe):**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Debian/Ubuntu: jeśli `venv` nie działa, doinstaluj go: `sudo apt install python3-venv`.

Po aktywacji na początku linii terminala pojawi się `(.venv)` — oznacza to, że środowisko
jest aktywne i polecenia `python` / `pip` działają w jego obrębie.

**Praca ze środowiskiem na co dzień:**

| Czynność | Windows | Linux / macOS |
|---|---|---|
| Włączenie (w każdym nowym terminalu) | `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| Wyłączenie | `deactivate` | `deactivate` |
| Usunięcie środowiska | usuń folder `.venv` | `rm -rf .venv` |

Biblioteki instaluje się **tylko raz**. Przy kolejnych uruchomieniach wystarczy aktywować
środowisko i uruchomić skrypt.

### Wariant 2: bez środowiska wirtualnego (instalacja globalna)

```bash
python -m pip install --user -r requirements.txt       # Windows
python3 -m pip install --user -r requirements.txt      # Linux / macOS
```

> Na nowszych dystrybucjach Linuksa (Ubuntu 23.04+, Debian 12+) instalacja globalna jest
> blokowana błędem `externally-managed-environment` — użyj wtedy wariantu 1.

### Wariant 3: Conda / Miniconda (jeśli już jej używasz)

```bash
conda create -n scansplit python=3.12 -y
conda activate scansplit
python -m pip install -r requirements.txt
```

### Sprawdzenie instalacji

```bash
python -c "import cv2, numpy; print('OpenCV', cv2.__version__, '| NumPy', numpy.__version__)"
python scan_autosplitter.py --help
```

Jeśli oba polecenia działają bez błędów — wszystko gotowe.

---

## 4. Uruchomienie

Podstawowe użycie: podaj folder ze skanami i folder na wyniki (zostanie utworzony automatycznie).

```bash
python scan_autosplitter.py ŚCIEŻKA_DO_SKANÓW ŚCIEŻKA_DO_WYNIKÓW
```

Przykłady:

```bash
# Linux / macOS
python scan_autosplitter.py ~/Skany/album ~/Skany/album_wyniki

# Windows (ścieżki ze spacjami w cudzysłowie)
python scan_autosplitter.py "C:\Users\Jan\Skany\album" "C:\Users\Jan\Skany\wyniki"
```

### Opcje

| Opcja | Opis |
|---|---|
| `--format png` / `--format tiff` | format wyjściowy (domyślnie `png`; `tiff` = bez kompresji) |
| `--workers N` | liczba równoległych procesów (domyślnie: liczba rdzeni − 1) |
| `--recursive` | przetwarzaj także podfoldery |
| `--debug` | zapisz podgląd detekcji `<skan>_debug.jpg` (ramki na skanie + maska) |
| `--no-face-rotate` | wyłącz automatyczne obracanie zdjęć wg twarzy |
| `-v`, `--verbose` | szczegółowe logi |

```bash
# TIFF, 4 procesy, z podglądem detekcji
python scan_autosplitter.py skany wyniki --format tiff --workers 4 --debug
```

### Wyniki

- Pliki nazywane są `[nazwa_skanu]_crop_01.png`, `_crop_02.png`, … w kolejności czytania
  (rzędami od góry, w rzędzie od lewej).
- Skan, na którym nic nie wykryto, jest zgłaszany ostrzeżeniem; uszkodzony plik — błędem,
  ale nie przerywa przetwarzania pozostałych.
- Na końcu wyświetla się podsumowanie, np. `Gotowe: 37 zdjęć z 12 skanów w 18.4s (błędy: 0)`.

---

## 5. Strojenie parametrów

Wszystkie parametry są stałymi na początku pliku `scan_autosplitter.py` (sekcja
`KONFIGURACJA`). Najpierw uruchom kilka skanów z `--debug` i obejrzyj pliki `*_debug.jpg`.

| Problem | Co zmienić |
|---|---|
| Wykrywa kurz / drobne śmieci jako zdjęcia | zwiększ `MIN_AREA_FRAC` |
| Pomija małe zdjęcia (np. legitymacyjne) | zmniejsz `MIN_AREA_FRAC` i `MIN_SIDE_FRAC` |
| Jasne zdjęcia nie są wykrywane / są ucięte | zmniejsz `BG_DIFF_THRESHOLD` (np. 10) oraz `CANNY_LOW`/`CANNY_HIGH` (np. 20/60) |
| Tło skanera (szum, przebarwienia) traktowane jak zdjęcie | zwiększ `BG_DIFF_THRESHOLD` (np. 18–20) |
| Ciemna ramka skanera dokleja się do zdjęć (ramka obejmuje pół skanu) | zmniejsz `SCANNER_EDGE_LINE_FRAC` (np. 0.85) |
| Zdjęcie dosunięte do krawędzi skanu jest ucięte | zwiększ `SCANNER_EDGE_LINE_FRAC` (np. 0.99) |
| Białe trójkąty / paski w rogach wycinków | zwiększ `CROP_INSET_PX` (np. 4–6) |
| Ucina białą ramkę odbitek | zmniejsz `GROW_DELTA` (np. 2.0) |
| Do zdjęcia doklejony pas tła | zwiększ `GROW_DELTA` (np. 5.0) lub ustaw `GROW_TO_PAPER_EDGE = False` |
| Dwa blisko leżące zdjęcia wycięte jako jedno | zmniejsz `CLOSE_KERNEL_FRAC` (np. 0.002) |
| Zdjęcie bez twarzy obrócone niepotrzebnie | zwiększ `FACE_MIN_CONFIDENCE` (np. 0.9) lub użyj `--no-face-rotate` |
| Zdjęcie z twarzami nie zostało obrócone | zmniejsz `FACE_MIN_CONFIDENCE` (np. 0.7) |

**Wskazówka przy skanowaniu:** zostawiaj choćby 2–3 mm odstępu między odbitkami — detekcja
jest wtedy najdokładniejsza. Zdjęcia nachodzące na siebie (bez widocznej przerwy) mogą zostać
wycięte jako jedno.

**Orientacja:** obracane są tylko zdjęcia, na których wykryto twarz. Zdjęcia bez ludzi
(krajobrazy, budynki) zostają w orientacji, w jakiej leżały na skanerze.

---

## 6. Rozwiązywanie problemów

| Komunikat | Rozwiązanie |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` / `Brak biblioteki: cv2` | biblioteki nie są zainstalowane, nie aktywowano `.venv` albo zainstalowano je do innego Pythona — patrz niżej „Kilka Pythonów na Windows” |
| `python -m venv .venv` → `Permission denied: 'C:\Program Files\LibreOffice\…'` | polecenie `python` uruchamia Pythona wbudowanego w LibreOffice — patrz niżej |
| `'python' is not recognized…` (Windows) | zainstaluj Pythona z opcją „Add to PATH” lub użyj `py` zamiast `python` |
| `externally-managed-environment` (Linux) | użyj środowiska wirtualnego (wariant 1) |
| `nie można odczytać pliku` | plik jest uszkodzony lub to nie jest obraz |
| Brak wyników dla skanu (`zdjęć: 0`) | uruchom z `--debug` i dostrój parametry (sekcja 5) |

---

### Kilka Pythonów na Windows (np. LibreOffice, Inkscape, GIMP, QGIS)

Niektóre programy instalują własnego Pythona i dopisują go do `PATH`. Wtedy polecenie
`python` może uruchamiać np. Pythona z LibreOffice (nie da się w nim utworzyć `venv`
ani zainstalować bibliotek), a `pip` — zupełnie innego Pythona.

1. Sprawdź, które Pythony są widoczne (pierwszy na liście wygrywa):
   ```bat
   where python
   where pip
   py -0p
   ```
2. Utwórz środowisko **wskazując właściwego Pythona wprost** — wybierz działający wariant:
   ```bat
   py -3.13 -m venv .venv
   ```
   ```bat
   python3.13 -m venv .venv
   ```
   ```bat
   "%LOCALAPPDATA%\Microsoft\WindowsApps\python3.13.exe" -m venv .venv
   ```
   (ostatni wariant dotyczy Pythona z Microsoft Store; dla instalacji z python.org ścieżka to
   zwykle `"%LOCALAPPDATA%\Programs\Python\Python313\python.exe"`).
3. Aktywuj środowisko i zainstaluj biblioteki — po aktywacji `python` wskazuje już na `.venv`:
   ```bat
   .venv\Scripts\activate.bat
   python -m pip install -r requirements.txt
   python scan_autosplitter.py "D:\skany" "D:\skany\wyniki"
   ```

Bez środowiska wirtualnego wystarczy konsekwentnie używać tego samego interpretera do
instalacji i uruchamiania, np. `python3.13 -m pip install -r requirements.txt`, a potem
`python3.13 scan_autosplitter.py …`.

Na stałe można też usunąć katalog LibreOffice z `PATH` (Ustawienia → System → Informacje →
Zaawansowane ustawienia systemu → Zmienne środowiskowe) lub przesunąć właściwego Pythona wyżej.

---

## 7. Jak to działa (w skrócie)

1. Wczytanie skanu z zachowaniem głębi bitowej.
2. Detekcja na pomniejszonej kopii: kolor tła skanera (dominujący jasny, neutralny odcień),
   mapa odległości od tła (Lab) + krawędzie Canny, usunięcie ramki/cienia skanera, operacje
   morfologiczne, rozdzielanie sklejonych zdjęć (transformata odległości + watershed),
   filtracja konturów (pole, wymiary, prostokątność).
3. Doprecyzowanie każdej ramki na pełnej rozdzielczości i dołączenie białej ramki odbitki
   (uśrednione profile linii wzdłuż każdego boku).
4. Prostowanie (transformacja perspektywiczna, interpolacja Lanczos). Zdjęcia leżące prosto
   (kąt < 0.05°) są tylko wycinane — piksele identyczne z oryginałem.
5. Orientacja wg twarzy: detektor YuNet sprawdza 4 obroty; zdjęcie jest obracane, gdy
   w innej orientacji twarz jest wykrywana wyraźnie pewniej (punkty oczu/nosa/ust muszą
   układać się pionowo).
6. Zapis bezstratny: PNG lub nieskompresowany TIFF.
