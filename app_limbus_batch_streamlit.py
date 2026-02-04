# =========================
# ENV FIXES (IMPORTANT)
# =========================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# =========================
# IMPORTS
# =========================
import streamlit as st
import sys
import cv2
import numpy as np
import torch
from PIL import Image
import io
import zipfile

# =========================
# PATH FIX
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# =========================
# SAFE IMPORT
# =========================
try:
    from inference_utils import load_model, predict_masks
except ImportError as e:
    st.error("❌ Could not import inference_utils.py")
    st.error(str(e))
    st.write("Files in current directory:")
    st.write(os.listdir(CURRENT_DIR))
    st.stop()

# =========================
# CONFIG
# =========================
DEFAULT_MODEL_PATH = os.path.join(
    CURRENT_DIR, "model_limbus_crop_unetpp_weighted.pth"
)

# =========================
# HELPERS
# =========================
def crop_limbus(image_bgr, mask):
    """Apply mask + crop bounding box."""
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return masked

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return masked[y:y+h, x:x+w]


def image_to_bytes(img_rgb):
    """Convert numpy RGB image to PNG bytes."""
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_resource
def get_model(model_path, device):
    if not os.path.exists(model_path):
        return None, None, None, None
    return load_model(model_path, device)

# =========================
# MAIN UI
# =========================
def main():
    st.set_page_config(
        page_title="Limbus Cropper",
        layout="wide"
    )

    st.title("👁️ Limbus Extraction & Cropping Tool")
    st.caption("Automatic limbus segmentation + crop | Single & Batch Download")

    # -------------------------
    # SIDEBAR
    # -------------------------
    st.sidebar.header("⚙️ Configuration")

    model_path = st.sidebar.text_input(
        "Model Path",
        value=DEFAULT_MODEL_PATH
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.write(f"**Device:** `{device}`")

    # -------------------------
    # LOAD MODEL
    # -------------------------
    with st.spinner("Loading model..."):
        model, idx_crop, idx_limbus, img_size = get_model(
            model_path, device
        )

    if model is None:
        st.error("❌ Model not found. Check path.")
        st.stop()

    st.sidebar.success("✅ Model loaded")

    # -------------------------
    # UPLOAD
    # -------------------------
    uploaded_files = st.file_uploader(
        "📤 Upload Eye Images",
        type=["jpg", "jpeg", "png", "bmp", "tif"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload images to begin.")
        return

    st.write(f"### Processing {len(uploaded_files)} image(s)")

    processed = []
    progress = st.progress(0)
    cols = st.columns(3)

    # -------------------------
    # PROCESS LOOP
    # -------------------------
    for i, file in enumerate(uploaded_files):
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, 1)

        if bgr is None:
            continue

        masks = predict_masks(model, bgr, img_size, device)

        limbus_mask = masks[idx_limbus]
        limbus_mask = (limbus_mask > 0.5).astype(np.uint8)

        cropped = crop_limbus(bgr, limbus_mask)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        fname = os.path.splitext(file.name)[0] + "_limbus.png"
        processed.append((fname, rgb))

        # DISPLAY
        if i < 9:
            with cols[i % 3]:
                st.image(rgb, caption=file.name, use_container_width=True)

                st.download_button(
                    "⬇️ Download",
                    data=image_to_bytes(rgb),
                    file_name=fname,
                    mime="image/png",
                    key=f"single_{i}"
                )

        progress.progress((i + 1) / len(uploaded_files))

    st.success("✅ Processing complete")

    # -------------------------
    # SINGLE IMAGE DOWNLOAD
    # -------------------------
    if len(processed) == 1:
        fname, img = processed[0]
        st.download_button(
            "📥 Download Cropped Image",
            data=image_to_bytes(img),
            file_name=fname,
            mime="image/png"
        )

    # -------------------------
    # ZIP DOWNLOAD
    # -------------------------
    if len(processed) > 1:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for fname, img in processed:
                zf.writestr(fname, image_to_bytes(img))

        st.download_button(
            "📦 Download All (ZIP)",
            data=zip_buf.getvalue(),
            file_name="limbus_crops.zip",
            mime="application/zip"
        )


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
