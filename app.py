import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
from lego_mosaic import quantize_image_to_lego, lego_palette, count_studs, STUD_SIZE_MM
from stud_overlay import add_studs_overlay, print_dimensions_and_examples

st.title("🧱 LEGO Mosaic Generator")

# Upload input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Sidebar options
st.sidebar.header("Settings")
studs_w = st.sidebar.number_input("Width (studs)", min_value=20, max_value=500, value=100)
studs_h = st.sidebar.number_input("Height (studs)", min_value=20, max_value=500, value=100)
quality = st.sidebar.selectbox("Quality", ["low", "medium", "high"])
stud_type = st.sidebar.selectbox("Stud Type", ["round", "flat", "technic"])
block_size = st.sidebar.slider("Block size (visual pixels per stud)", 10, 40, 20)
stud_ratio = st.sidebar.slider("Stud ratio", 0.5, 1.0, 0.75)

if uploaded_file:
    # Load image and resize to studs
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((studs_w, studs_h), Image.LANCZOS)

    # Quality → number of colors
    quality_map = {"low": 3, "medium": 12, "high": len(lego_palette)}
    n_colors = quality_map[quality]

    # Convert to LEGO palette
    lego_img = quantize_image_to_lego(img_resized, lego_palette)

    # Overlay studs
    overlay_img, color_counter, w, h = add_studs_overlay(
        lego_img,
        block_size=block_size,
        stud_ratio=stud_ratio,
        stud_type=stud_type
    )

    # Show result
    st.image(overlay_img, caption="LEGO Mosaic", use_column_width=True)

    # Studs per color
    st.subheader("Studs Required per Color")
    for color, count in color_counter.items():
        if count > 0:
            st.write(f"Color {color}: {count} studs")

    # Real-life size
    st.subheader("Real-life Dimensions")
    width_mm = w * STUD_SIZE_MM
    height_mm = h * STUD_SIZE_MM
    width_m = width_mm / 1000
    height_m = height_mm / 1000
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4

    st.write(f"{width_m:.2f}m x {height_m:.2f}m  ({width_in:.1f}in x {height_in:.1f}in)")

    # Download button
    st.download_button(
        "Download Mosaic Image",
        data=overlay_img.tobytes(),
        file_name="lego_mosaic.png",
        mime="image/png"
    )
