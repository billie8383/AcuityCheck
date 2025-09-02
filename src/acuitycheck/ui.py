from pathlib import Path
from typing import List
import numpy as np
# import streamlit as st
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

TITLE = "AcuityCheck"
MODEL_DIR = Path("models/onnx")
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"


def ui_intro(st) -> None:
    # st.title(TITLE)
    ui_logo(st)
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

def ui_logo(st):
    left, mid, right = st.columns([1,1,1])
    with mid:
        st.image("static/logo.png", use_container_width=True)

def ui_models_panel(model_path: Path ,st):
    """Show model presence and simple instructions. Returns True if YuNet exists."""
    models_ready = model_path.exists()
  
    if models_ready:
        st.success("OpenCV YuNet model found.")
    else:
        st.warning("YuNet model not found. Run scripts/download_models.sh")

def sidebar_settings(st):
    st.sidebar.header("Settings")

    with st.sidebar.expander("Detection", expanded=False):
        ui_models_panel(YUNET_PATH, st)
        det_thresh = st.select_slider(
            "Detector score threshold",
            options=[round(x, 2) for x in np.arange(0.05, 0.95, 0.05)],
            value=0.30,
            help="Lower if detection struggles; raise to avoid false positives."
        )
        st.caption("YuNet runs via OpenCV FaceDetectorYN. Ensure the ONNX model file exists.")

    with st.sidebar.expander("Distance inputs", expanded=True):
        ipd_mm = st.number_input("Interpupillary distance (mm)", 40, 80, 63, 1)
        offset_mm = st.number_input("Camera to screen offset (mm)", 0, 150, 40, 1)
        known_mm = st.number_input("Camera to eye distance (mm) for calibration", 200, 2000, 500, 10,
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

def build_chart_lines(
    chart_style: str,
    denominators: List[int],
    single_letter: str = "A",
) -> List[str]:
    """Builds optotype lines for a vision chart.

    Supports multiple chart styles:
        - "classic snellen": Standard 8-line Snellen chart.
        - "single letter": Repeated single letter, increasing length per line.
        - Default: Sloan letter mix ("CDHKNORSVZ") rotated per line.

    Args:
        chart_style (str): Chart style ("classic snellen", "single letter", or default Sloan).
        denominators (List[int]): List of Snellen denominators, one per line.
        single_letter (str, optional): Letter used for the "single letter" style. 
            Defaults to "A".

    Returns:
        List[str]: List of strings, one per chart line.
    """
    chart_style = chart_style.lower()

    if chart_style == "classic snellen":
        # 8 standard lines, top to bottom. Uses common Snellen optotypes.
        classic_snellen = [
            "E",
            "FP",
            "TOZ",
            "LPED",
            "PECFD",
            "EDFCZP",
            "FELOPZD", 
            "DEFPOTEC",
        ]
        # Adjust number of lines based on denominators
        if len(denominators) <= len(classic_snellen):
            return classic_snellen[: len(denominators)]
        else:
            # Extend with repeats of the last line
            return classic_snellen + [classic_snellen[-1]] * (len(denominators) - len(classic_snellen))

    if chart_style == "single letter":
        # Each line grows in length, clamped between 2 and 10 letters
        return [single_letter * max(2, min(10, 2 + i)) for i in range(len(denominators))]

    # Default: Sloan mix (CDHKNORSVZ), rotated per line
    base_letters = list("CDHKNORSVZ")
    chart_lines: List[str] = []
    for i in range(len(denominators)):
        n_letters = max(2, min(10, 2 + i))  # number of letters in line
        letters = "".join(base_letters[(j + i) % len(base_letters)] for j in range(n_letters))
        chart_lines.append(letters)

    return chart_lines

def render_chart(letter_px_sizes, letters_per_line, show_labels, polarity, letter_spacing_em, st):
    fg  = "#1f2430" if polarity == "Dark on light" else "#f4f6fa"
    sub = "#9aa3ad" if polarity == "Dark on light" else "#b8c4d1"
    bg  = "#ffffff" if polarity == "Dark on light" else "#0b2733"

    # centre the whole block 
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            f"""
            <style>
              .snellen-wrap {{
                --linew: 70%;                 
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
                width:64px;                   
                text-align:right;
                font-size:0.95rem;
                color:{sub};
                visibility:{'visible' if show_labels else 'hidden'};
              }}
              .snellen-letters {{
                width: var(--linew);          
                margin: 0 auto;
                display: grid;
                column-gap: 0;                
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
                width: var(--linew);          
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

            # optional guide bars 
            if label == "6/12":
                st.markdown("<div class='snellen-bar-green'></div>", unsafe_allow_html=True)
            if label == "6/9":
                st.markdown("<div class='snellen-bar-red'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

