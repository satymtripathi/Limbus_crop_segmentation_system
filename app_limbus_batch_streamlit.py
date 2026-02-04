import streamlit as st
import cv2
import numpy as np
import torch
import os
import zipfile
import io
from PIL import Image

from inference_utils import (
    load_model,
    predict_masks,
    draw_smooth_contours
)

# =======================
# CONFIG
# =======================
MODEL_PATH = "model_limbus_crop_unetpp_weighted.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(layout="wide", page_title="Limbus Batch Segmentation")

# =======================
# LOAD MODEL (CACHED)
# =======================
@st.cache_resource
def load_segmentation_model():
    return load_model(MODEL_PATH, DEVICE)

model, idx_crop, idx_limbus, img_size = load_segmentation_model()

# =======================
# LIMBUS TIGHT CROP
# =======================
def crop_limbus_tightly(image_bgr, limbus_mask, pad=5):
    mask = (limbus_mask > 0).astype(np.uint8)

    if mask.sum() == 0:
        return None

    ys, xs = np.where(mask > 0)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    h, w = image_bgr.shape[:2]
    y1 = max(0, y1 - pad)
    y2 = min(h, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(w, x2 + pad)

    crop = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    crop[crop_mask == 0] = 0
    return crop

# =======================
# PROCESS IMAGE
# =======================
def process_single_image(image_bgr):
    masks = predict_masks(model, image_bgr, img_size, DEVICE)
    limbus_mask = masks[idx_limbus]

    overlay = draw_smooth_contours(image_bgr, None, limbus_mask)
    limbus_crop = crop_limbus_tightly(image_bgr, limbus_mask)

    return overlay, limbus_crop

# =======================
# ZIP CREATOR
# =======================
def create_zip_from_folder(folder_path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer

# =======================
# PASSPORT GRID
# =======================
def render_passport_grid(image_paths, thumb_size=96, cols=6):
    if len(image_paths) == 0:
        st.info("No limbus crops to preview")
        return

    rows = (len(image_paths) + cols - 1) // cols
    for r in range(rows):
        row_imgs = image_paths[r * cols:(r + 1) * cols]
        columns = st.columns(cols)

        for col, img_path in zip(columns, row_imgs):
            img = Image.open(img_path).convert("RGB")
            img = img.resize((thumb_size, thumb_size))
            col.image(img)
            col.caption(os.path.basename(img_path))

# =======================
# UI
# =======================
st.title("👁️ Limbus Segmentation – Batch Processor")
st.markdown("**Tight limbus-only cropping · Passport preview · ZIP download**")

mode = st.radio("Select Mode", ["Single Image", "Batch Folder"])

# =======================
# SINGLE IMAGE MODE
# =======================
if mode == "Single Image":
    uploaded = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
        overlay, limbus_crop = process_single_image(image)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original")

        with c2:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Limbus Overlay")

        with c3:
            if limbus_crop is not None:
                st.image(cv2.cvtColor(limbus_crop, cv2.COLOR_BGR2RGB),
                         caption="Tight Limbus Crop")
            else:
                st.warning("No limbus detected")

# =======================
# BATCH MODE
# =======================
else:
    input_dir = st.text_input("Input Folder Path")
    output_dir = st.text_input("Output Folder Path", "limbus_outputs")

    show_grid = st.checkbox("Show passport grid preview", value=True)

    if st.button("Run Batch Processing"):
        os.makedirs(output_dir, exist_ok=True)

        images = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        progress = st.progress(0)
        status = st.empty()

        limbus_paths = []

        for i, img_name in enumerate(images):
            img_path = os.path.join(input_dir, img_name)
            image = cv2.imread(img_path)

            overlay, limbus_crop = process_single_image(image)
            base = os.path.splitext(img_name)[0]

            if overlay is not None:
                cv2.imwrite(
                    os.path.join(output_dir, f"{base}_overlay.png"),
                    overlay
                )

            if limbus_crop is not None:
                limbus_path = os.path.join(output_dir, f"{base}_limbus.png")
                cv2.imwrite(limbus_path, limbus_crop)
                limbus_paths.append(limbus_path)

            progress.progress((i + 1) / len(images))
            status.text(f"Processed {i + 1}/{len(images)}")

        st.success("✅ Batch processing completed")

        # GRID PREVIEW
        if show_grid:
            st.markdown("### 🪪 Limbus Passport Preview")
            render_passport_grid(limbus_paths)

        # ZIP DOWNLOAD
        zip_buffer = create_zip_from_folder(output_dir)
        st.download_button(
            label="⬇️ Download ALL Results (ZIP)",
            data=zip_buffer,
            file_name="limbus_batch_outputs.zip",
            mime="application/zip"
        )

