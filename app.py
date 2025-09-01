import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import streamlit as st
from PIL import Image
import cv2

# Make src importable when running via `streamlit run` from repo root
SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from acuitycheck import CARD_W_MM, CARD_H_MM  
from acuitycheck.detection import run_yunet_cv2, detect_eyes_in_roi 
from acuitycheck.geometry import compute_distance_mm, fov_from_fpx  
from acuitycheck.snellen import snellen_letter_height_mm  


TITLE = "AcuityCheck"
MODEL_DIR = Path("models/onnx")
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

# --- session defaults ---
if "f_px" not in st.session_state:
    st.session_state.f_px = None
if "eye_to_screen_mm" not in st.session_state:
    st.session_state.eye_to_screen_mm = None


# ---------- helpers ----------
def draw_debug_overlay(
    frame_bgr: np.ndarray,
    box: Optional[Tuple[float, float, float, float]],
    pts: Optional[List[Tuple[float, float]]],
) -> Image.Image:
    img = frame_bgr.copy()
    if box is not None:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 255), 2)
    if pts:
        for (px, py) in pts:
            cv2.circle(img, (int(px), int(py)), 2, (0, 255, 0), -1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ---------- UI sections as functions ----------
def ui_intro() -> None:
    # st.title(TITLE)
    ui_logo()
    st.caption("Indicative only — not a medical device.")

    with st.expander("How this works?", expanded=False):
        st.markdown(
            """
    **How this works (quick steps)**  
    1) **Calibrate screen size** with a real credit/debit card.  
    2) **Take a snapshot**; we measure your **pixel IPD** (eye spacing in pixels).  
    3) **One-off calibration:** enter a known **camera→eye distance** to compute the camera’s **focal length (px)**.  
    4) We compute your **eye→screen distance** (subtracting the camera→screen offset) and size the Snellen chart automatically.

    """
        )


def ui_models_panel(model_path: Path):
    """Show model presence and simple instructions. Returns True if YuNet exists."""
    models_ready = model_path.exists()
  
    if models_ready:
        st.success("OpenCV YuNet model found.")
    else:
        st.warning("YuNet model not found. Run scripts/download_models.sh")


def build_chart_lines(chart_style: str, denominators: list[int], single_letter: str = "A") -> list[str]:
    """
    Returns list of strings (one per line), matching the length of denominators.
    """
    chart_style = chart_style.lower()
    if chart_style == "classic snellen":
        # 8 standard lines, top to bottom. Uses common Snellen optotypes.
        classic = [
            "E",
            "FP",
            "TOZ",
            "LPED",
            "PECFD",
            "EDFCZP",
            "FEL0PZD".replace("0","O"),  # ensure letter O, not zero
            "DEFPOTEC",
        ]
        # If you have more/less denominators, truncate or extend gracefully
        if len(denominators) <= len(classic):
            return classic[:len(denominators)]
        else:
            # Repeat last line if more levels are requested
            return classic + [classic[-1]] * (len(denominators) - len(classic))

    if chart_style == "single letter":
        return [single_letter * max(2, min(10, 2 + i)) for i in range(len(denominators))]

    # Default: Sloan mix (CDHKNORSVZ), rotating letters per line
    base_letters = list("CDHKNORSVZ")
    lines = []
    for i in range(len(denominators)):
        n = max(2, min(10, 2 + i))
        letters = "".join(base_letters[(j + i) % len(base_letters)] for j in range(n))
        lines.append(letters)
    return lines


def sidebar_settings():
    st.sidebar.header("Settings")

    with st.sidebar.expander("Detection", expanded=False):
        ui_models_panel(YUNET_PATH)
        det_thresh = st.select_slider(
            "Detector score threshold",
            options=[round(x, 2) for x in np.arange(0.05, 0.95, 0.05)],
            value=0.30,
            help="Lower if detection struggles; raise to avoid false positives."
        )
        st.caption("YuNet runs via OpenCV FaceDetectorYN. Ensure the ONNX model file exists.")

    with st.sidebar.expander("Distance inputs", expanded=True):
        ipd_mm = st.number_input("Interpupillary distance (mm)", 40, 80, 63, 1)
        offset_mm = st.number_input("Camera → screen offset (mm)", 0, 150, 40, 1)
        known_mm = st.number_input("Known camera → eye distance (mm) for calibration", 200, 2000, 500, 10,
                                   help="One-off: sit at a measured distance then calibrate f.")

    with st.sidebar.expander("Chart", expanded=True):
        chart_style = st.selectbox(
            "Chart style",
            ["Sloan mix", "Classic Snellen", "Single letter"]  # ← added Classic Snellen here
        )
        single_letter = st.text_input("Single letter", value="A", max_chars=1).strip().upper() or "A"
        polarity = st.selectbox("Polarity", ["Dark on light", "Light on dark"])
        letter_spacing_em = st.slider("Letter spacing (em)", 0.00, 0.20, 0.05, 0.01)
        show_labels = st.checkbox("Show 6/x labels", value=True)


    return {
        "det_thresh": float(det_thresh),
        "ipd_mm": float(ipd_mm),
        "offset_mm": float(offset_mm),
        "known_mm": float(known_mm),
        "chart_style": chart_style,
        "single_letter": single_letter,
        "polarity": polarity,
        "letter_spacing_em": float(letter_spacing_em),
        "show_labels": bool(show_labels),
    }

def render_chart(letter_px_sizes, letters_per_line, *, show_labels, polarity, letter_spacing_em):
    fg  = "#1f2430" if polarity == "Dark on light" else "#f4f6fa"
    sub = "#9aa3ad" if polarity == "Dark on light" else "#b8c4d1"
    bg  = "#ffffff" if polarity == "Dark on light" else "#0b2733"

    # centre the whole block using columns so it plays nicely with Streamlit
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            f"""
            <style>
              .snellen-wrap {{
                --linew: 70%;                 /* <<< constant line width (tweak to taste) */
                background:{bg};
                padding:20px 0;
                border-radius:12px;
                max-width: 720px;
                margin: 0 auto;
              }}
              .snellen-row {{
                display:flex;
                align-items:center;
                gap:12px;
                margin:26px 0;
              }}
              .snellen-label {{
                width:64px;                   /* fixed left gutter for 6/x */
                text-align:right;
                font-size:0.95rem;
                color:{sub};
                visibility:{'visible' if show_labels else 'hidden'};
              }}
              .snellen-letters {{
                width: var(--linew);          /* <<< same left/right edges for all rows */
                margin: 0 auto;
                display: grid;
                column-gap: 0;                /* spacing handled by letter-spacing below */
                justify-items: center;
              }}
              .snellen-letters span {{
                letter-spacing:{letter_spacing_em:.2f}em;
                color:{fg};
                font-weight:700;
                font-family: 'Arial Black', ui-sans-serif, system-ui;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
              }}
              .snellen-bar-green, .snellen-bar-red {{
                height:6px;
                width: var(--linew);          /* bars align with letter block width */
                margin: 18px auto 4px auto;
                border-radius:3px;
              }}
              .snellen-bar-green {{ background:#2ecc71; }}
              .snellen-bar-red   {{ background:#e74c3c; }}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='snellen-wrap'>", unsafe_allow_html=True)

        for (label, px), line in zip(letter_px_sizes, letters_per_line):
            n = max(1, len(line))
            # grid with n equal columns → letters align flush left & right for every row
            letter_spans = "".join(
                f"<span style='font-size:{px:.2f}px'>{ch}</span>" for ch in line
            )
            st.markdown(
                f"""
                <div class='snellen-row'>
                  <div class='snellen-label'>{label}</div>
                  <div class='snellen-letters' style='grid-template-columns: repeat({n}, 1fr);'>
                    {letter_spans}
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # optional guide bars (kept aligned to the same width)
            if label == "6/12":
                st.markdown("<div class='snellen-bar-green'></div>", unsafe_allow_html=True)
            if label == "6/9":
                st.markdown("<div class='snellen-bar-red'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def ui_logo():
    left, mid, right = st.columns([1,2,1])
    with mid:
        st.image("static/logo.png", width=True)


# ---------- main ----------
def main():
    st.set_page_config(page_title=TITLE, layout="centered")
    ui_intro()
 


    # Sidebar “dropdown” settings
    S = sidebar_settings()

    # Shared session defaults
    if "f_px" not in st.session_state: st.session_state.f_px = None
    if "eye_to_screen_mm" not in st.session_state: st.session_state.eye_to_screen_mm = None

    # Model presence panel (short)
    models_ready = YUNET_PATH.exists()

  
 
    # TABS = steps
    tab1, tab2, tab3 = st.tabs(["Step 1. Card calibration", "Step 2. Detection & calibration", "Step 3. Chart"])

    # ---- Step 1: Card calibration ----
    with tab1:
        st.subheader("Calibrate screen with a real card")
        width_px = st.slider("Adjust card width (px) to match your card", 80, 600, 220, 1)
        card_ppm = width_px / CARD_W_MM
        card_h_px = width_px * (CARD_H_MM / CARD_W_MM)
        st.markdown(
            f"<div style='width:{width_px}px;height:{card_h_px:.0f}px;border:2px solid #2bb3ff;background:#0b2733;'></div>",
            unsafe_allow_html=True
        )
        st.caption("Tip: set browser zoom to 100%.")
        st.info(f"Pixels per millimetre: **{card_ppm:.3f} px/mm**")
        st.session_state.card_ppm = float(card_ppm)

    # ---- Step 2: Detection & focal calibration ----
    with tab2:
        st.subheader("Take a snapshot")
        snap = st.camera_input("Face the camera; ensure good lighting.")
        pixel_ipd = None
        frame_bgr = None

        if snap is not None and models_ready:
            img = Image.open(snap)
            frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            box, det_kps = run_yunet_cv2(frame_bgr, model_path=YUNET_PATH, score_thresh=S["det_thresh"])
            pts = None
            if det_kps and len(det_kps) >= 2:
                le, re = sorted(det_kps[:2], key=lambda p: p[0])
                pixel_ipd = float(np.hypot(le[0]-re[0], le[1]-re[1]))
                pts = det_kps
            elif box is not None:
                eyes = detect_eyes_in_roi(frame_bgr, box)
                if eyes and len(eyes) >= 2:
                    le, re = eyes[0], eyes[1]
                    pixel_ipd = float(np.hypot(le[0]-re[0], le[1]-re[1]))
                    pts = eyes

            if pixel_ipd is None:
                st.error("Could not estimate eyes. Try brighter lighting and face the camera.")
            else:
                st.success(f"Pixel IPD: {pixel_ipd:.1f} px")

            if frame_bgr is not None:
                st.image(draw_debug_overlay(frame_bgr, box, pts), caption="Detection & landmarks")

            # Focal calibration
            st.markdown("#### Focal calibration")
            if st.button("Calibrate f (uses latest pixel IPD)"):
                if pixel_ipd is None:
                    st.warning("Take a snapshot first.")
                else:
                    st.session_state.f_px = (pixel_ipd * S["known_mm"]) / S["ipd_mm"]
                    st.success(f"Calibrated focal length: **{st.session_state.f_px:.1f} px**")

            # Live distance/FOV if calibrated
            if st.session_state.f_px and (pixel_ipd is not None) and (frame_bgr is not None):
                w_px, h_px = frame_bgr.shape[1], frame_bgr.shape[0]
                hfov, vfov, dfov = fov_from_fpx(float(st.session_state.f_px), w_px, h_px)
                _, eye_to_screen = compute_distance_mm(pixel_ipd, S["ipd_mm"], float(st.session_state.f_px), S["offset_mm"])
                st.session_state.eye_to_screen_mm = float(eye_to_screen)
                st.info(f"FOV ≈ **{hfov:.1f}° × {vfov:.1f}°** (diag {dfov:.1f}°) · Eye → screen: **{eye_to_screen:.0f} mm**")
            elif not st.session_state.f_px:
                st.warning("Not calibrated yet — enter a known distance in the sidebar and click **Calibrate f**.")

    # ---- Step 3: Chart (centred) ----
    with tab3:
        st.subheader("Snellen chart")
        st.caption("Sized from your measured eye → screen distance and card calibration.")

        card_ppm = float(st.session_state.get("card_ppm", 0)) or 0.0
        distance_mm = st.session_state.get("eye_to_screen_mm")
        if not distance_mm:  # covers None or 0
            distance_mm = 3000.0
        distance_mm = float(distance_mm)

        denominators = [60, 48, 36, 24, 18, 12, 9, 6]
        letter_lines = build_chart_lines(S["chart_style"], denominators, single_letter=S["single_letter"])
        distance_mm = float(st.session_state.get("eye_to_screen_mm") or 3000.0)
    
       
        # Convert to pixel sizes
        letter_px_sizes = []
        for den in denominators:
            mm = snellen_letter_height_mm(distance_mm, den)
            px = mm * card_ppm
            letter_px_sizes.append((f"6/{int(den)}", px))

        render_chart(
            letter_px_sizes,
            letter_lines,
            show_labels=S["show_labels"],
            polarity=S["polarity"],
            letter_spacing_em=S["letter_spacing_em"],
        )


if __name__ == "__main__":
    main()
