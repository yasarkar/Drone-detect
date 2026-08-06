"""
Interactive calibration helper for the GeoMapper (`geo_mapping`) configuration.

Run from the project root:
    python tools/calibrate.py

It walks you through measuring/entering the camera parameters and prints
a ready-to-paste YAML block for `config/config.yaml`.
"""

import math
import sys

BANNER = [
    "=" * 62,
    "  GeoMapper - Kamera Kalibrasyon Asistani",
    "  (Drone pixel -> gercek dunya GPS donusumu)",
    "=" * 62,
]

MEASUREMENT_STEPS = [
    "1) KAMERA GPS KONUMU  : Kameranin durdugu noktayi telefon haritandan oku",
    "   (Google Maps'e konuma uzun bas -> koordinatlar cikar)",
    "",
    "2) HEADING (Yaw)      : Pusula uygulamasiyla kameranin tam onune baktigi",
    "   yon. Kuzey=0, Dogu=90, Gney=180, Bati=270",
    "",
    "3) PITCH              : Kameranin optik ekseninin ufka gore acisi.",
    "   Asi-yi bakiyor = NEGATIF (orn. -10), yukari bakiyor = POZITIF",
    "",
    "4) FOV (Yatay)        : Asagidaki basit olcumle lensin yatay gorus acisi.",
    "",
]


def ask_float(prompt: str, default: float | None = None) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print(f"  Gecersiz sayi, tekrar dene. (orn: {default})")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [E/h]" if default else " [e/H]"
    while True:
        raw = input(f"{prompt}{suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in ("e", "evet", "y", "yes"):
            return True
        if raw in ("h", "hayir", "n", "no"):
            return False
        print("  E/H diye cevapla.")


def measure_fov_dialog() -> float:
    """Prompts for distance/width and computes FOV_h = 2*atan(W/(2D))."""
    print("\n--- YATAY FOV OLÇÜMÜ ---")
    print("Kamerayi bir duvara dik olarak yerlestir.")
    print("  D = Kameranin duvara olan uzakligi (metre)")
    print("  W = Goruntude soldan saga gorunen duvar genisligi (metre)")
    print("Ornek: D=5m, W=7m  ->  FOV = 2*atan(7/(2*5)) = 70 derece\n")

    d = ask_float("D (metre)", default=5.0)
    w = ask_float("W (metre)", default=7.0)

    if d <= 0:
        print("  D pozitif olmali. Varsayilan FOV kullaniliyor.")
        return 70.0

    fov = math.degrees(2.0 * math.atan(w / (2.0 * d)))
    print(f"\n  Hesaplanan FOV_h = {fov:.2f} derece")
    return fov


def build_yaml_block(cfg: dict) -> str:
    """Produces the geo_mapping YAML block."""
    cam = cfg["camera"]
    lines = [
        "geo_mapping:",
        f"  enabled: {str(cfg['enabled']).lower()}",
        "  camera:",
        f"    latitude: {cam['latitude']}",
        f"    longitude: {cam['longitude']}",
        f"    altitude_m: {cam['altitude_m']}",
        f"    heading_deg: {cam['heading_deg']}",
        f"    pitch_deg: {cam['pitch_deg']}",
        f"    roll_deg: {cam['roll_deg']}",
        f"    fov_h_deg: {cam['fov_h_deg']}",
        f"    fov_v_deg: 0.0",
        f"    reference_size: [{cam['ref_w']}, {cam['ref_h']}]",
        f"  target_altitude_amsl_m: {cfg['target_alt']}",
        f"  show_on_screen: {str(cfg['show_on_screen']).lower()}",
        "",
    ]
    return "\n".join(lines)


def main():
    for line in BANNER:
        print(line)

    print("\nÖLÇÜM REHBERI")
    for step in MEASUREMENT_STEPS:
        print(f"  {step}")

    print("\nSimdi degerleri gir.\n")

    # 1. Camera GPS
    print("--- 1) KAMERA GPS KONUMU ---")
    lat = ask_float("Kamera enlemi (latitude, orn. 41.0082)", default=41.0082)
    lon = ask_float("Kamera boylami (longitude, orn. 28.9784)", default=28.9784)
    alt = ask_float("Kamera deniz seviyesi yuksekligi (m)", default=5.0)

    # 2. Heading
    print("\n--- 2) HEADING (YAW) ---")
    print("  Kuzey=0, Dogu=90, Guney=180, Bati=270 derece")
    heading = ask_float("Kamera baktigi azimut (heading_deg)", default=0.0)

    # 3. Pitch
    print("\n--- 3) PITCH ---")
    print("  Astagi bakiyorsa NEGATIF (orn. -10), yukari bakiyorsa POZITIF")
    pitch = ask_float("Kamera egim acisi (pitch_deg)", default=-10.0)

    # 4. Roll
    print("\n--- 4) ROLL ---")
    roll = ask_float("Kamera yana yatma (roll_deg, cogu zaman 0)", default=0.0)

    # 5. FOV
    print("\n--- 5) YATAY FOV ---")
    print("  FOV'u biliyorsan direkt gir, bilmiyorsan (O)lcum ile bul.")
    raw = input("FOV_h biliniyor mu? (Evet enter ile, Hayir icin O yaz): ").strip().lower()
    if raw in ("o", "olcum", "hayir", "h", "no", "n"):
        fov_h = measure_fov_dialog()
    else:
        fov_h = ask_float("FOV_h (derece)", default=70.0)

    # 6. Reference resolution
    print("\n--- 6) REFERANS COZUNURLUK ---")
    print("  FOV degerlerinin olcumu hangi cozunurlukte yapildi?")
    ref_w = int(ask_float("Genislik (px)", default=1280))
    ref_h = int(ask_float("Yukseklik (px)", default=720))

    # 7. Target altitude
    print("\n--- 7) HEDEF (DRONE) YUKSEKLIGI ---")
    print("  Takip edilen drone'larin tipik ucus yuksekligi.")
    print("  target_alt = Bolgenin denizden yuksekligi + drone yerden yuksekligi")
    ground_alt = ask_float("Bolge deniz seviyesi yuksekligi (m)", default=40.0)
    drone_alt = ask_float("Drone yerden yuksekligi (m)", default=50.0)
    target_alt = round(ground_alt + drone_alt, 2)

    # 8. Show on screen
    print("\n--- 8) EKRAN GOSTERIMI ---")
    show_on_screen = ask_yes_no("Kutu etiketinin altinda Lat/Lon gosterilsin mi?", default=True)

    cfg = {
        "enabled": True,
        "camera": {
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "altitude_m": round(alt, 2),
            "heading_deg": round(heading, 2),
            "pitch_deg": round(pitch, 2),
            "roll_deg": round(roll, 2),
            "fov_h_deg": round(fov_h, 2),
            "ref_w": ref_w,
            "ref_h": ref_h,
        },
        "target_alt": target_alt,
        "show_on_screen": show_on_screen,
    }

    print("\n" + "=" * 62)
    print("Hazir config bloğu (config/config.yaml icindeki mevcut")
    print("geo_mapping blogunun yerine yapistir):")
    print("=" * 62)
    print()
    print(build_yaml_block(cfg))
    print("=" * 62)
    print("\nSonra dogrulamak icin:  python test_geo_mapping.py")
    print("Gerçek test: Bilinen bir yeri kameraya goster, ekrandaki")
    print("Lat/Lon degerlerini telefon GPS'in ile karsilastir.")
    print()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nIptal edildi.")
        sys.exit(1)