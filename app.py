import sys
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import cv2


SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from acuitycheck import CARD_W_MM, CARD_H_MM  
from acuitycheck.detection import run_yunet_cv2, detect_eyes_in_roi,draw_debug_overlay 
from acuitycheck.geometry import compute_distance_mm, fov_from_fpx  
from acuitycheck.snellen import snellen_letter_height_mm  
from acuitycheck.ui import (
    ui_intro,
    sidebar_settings,
    build_chart_lines,
    render_chart
)

TITLE = "AcuityCheck"
MODEL_DIR = Path("models/onnx")
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

# --- session defaults ---
if "f_px" not in st.session_state:
    st.session_state.f_px = None
if "eye_to_screen_mm" not in st.session_state:
    st.session_state.eye_to_screen_mm = None


# ---------- main ----------
def main():
    st.set_page_config(page_title=TITLE, layout="centered")
    ui_intro(st)
 
    # Sidebar “dropdown” settings
    S = sidebar_settings(st)

    # Shared session defaults
    if "f_px" not in st.session_state: st.session_state.f_px = None
    if "eye_to_screen_mm" not in st.session_state: st.session_state.eye_to_screen_mm = None

    # Model presence panel (short)
    models_ready = YUNET_PATH.exists()

 
    # TABS = steps
    tab1, tab2, tab3 = st.tabs(["Step 1. Card calibration", "Step 2. Detection & calibration", "Step 3. Chart"])

    # Step 1: Card calibration 
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

    # Step 2: Detection & focal calibration 
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

    # Step 3: Chart 
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
            st=st
        )


if __name__ == "__main__":
    main()
