import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from collections import Counter

# -------------------------------------------------
# LEGO Constants
# -------------------------------------------------
STUD_SIZE_MM = 8  # standard LEGO stud = 8mm

class LegoColor:
    def __init__(self, name, rgb):
        self.name = name
        self.rgb = rgb

lego_palette = [
    LegoColor("White", (242, 243, 242)),
    LegoColor("Black", (27, 42, 52)),
    LegoColor("Red", (196, 40, 28)),
    LegoColor("Dark Red", (123, 46, 47)),
    LegoColor("Blue", (0, 85, 191)),
    LegoColor("Dark Blue", (0, 38, 84)),
    LegoColor("Bright Blue", (31, 90, 171)),
    LegoColor("Yellow", (245, 205, 47)),
    LegoColor("Bright Yellow", (255, 205, 47)),
    LegoColor("Green", (75, 151, 74)),
    LegoColor("Dark Green", (0, 69, 26)),
    LegoColor("Bright Green", (88, 171, 65)),
    LegoColor("Tan", (215, 197, 153)),
    LegoColor("Dark Tan", (162, 140, 104)),
    LegoColor("Brown", (105, 64, 39)),
    LegoColor("Dark Brown", (73, 50, 35)),
    LegoColor("Light Gray", (160, 165, 169)),
    LegoColor("Dark Gray", (99, 95, 98)),
    LegoColor("Orange", (218, 133, 64)),
    LegoColor("Bright Orange", (214, 121, 35)),
    LegoColor("Purple", (123, 46, 129)),
    LegoColor("Magenta", (196, 98, 151)),
    LegoColor("Light Blue", (100, 180, 214)),
    LegoColor("Sand Green", (120, 144, 130)),
    LegoColor("Dark Tan Greenish", (116, 134, 117))
]

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def nearest_lego_color(pixel, palette):
    min_dist = float('inf')
    closest = palette[0].rgb
    for color in palette:
        dist = sum((int(p)-int(c))**2 for p,c in zip(pixel, color.rgb))
        if dist < min_dist:
            min_dist = dist
            closest = color.rgb
    return closest

def quantize_image_to_lego(img, palette):
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    for y in range(h):
        for x in range(w):
            arr[y, x] = np.array(nearest_lego_color(arr[y, x], palette), dtype=np.uint8)
    return Image.fromarray(arr)

def count_studs(img, palette):
    arr = np.array(img)
    counts = {color.name: 0 for color in palette}
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            px = tuple(arr[y, x])
            for color in palette:
                if px == color.rgb:
                    counts[color.name] += 1
    return counts

def add_studs_overlay(img, block_size=20, stud_ratio=0.75, stud_type="round"):
    arr = np.array(img)
    h, w, _ = arr.shape
    overlay_img = Image.new("RGB", (w * block_size, h * block_size), (255, 255, 255))
    draw = ImageDraw.Draw(overlay_img)
    color_counter = Counter()

    for y in range(h):
        for x in range(w):
            color = tuple(arr[y, x])
            color_counter[color] += 1
            x0, y0 = x * block_size, y * block_size
            x1, y1 = x0 + block_size, y0 + block_size
            draw.rectangle([x0, y0, x1, y1], fill=color)

            if stud_type != "flat":
                stud_d = block_size * stud_ratio
                stud_r = stud_d / 2
                cx, cy = x0 + block_size / 2, y0 + block_size / 2
                bbox = [cx - stud_r, cy - stud_r, cx + stud_r, cy + stud_r]

                if stud_type == "round":
                    highlight = tuple(min(255, int(c * 1.1)) for c in color)
                    draw.ellipse(bbox, fill=highlight, outline=(50, 50, 50))
                elif stud_type == "technic":
                    draw.ellipse(bbox, fill=color, outline=(50, 50, 50))
                    hole_r = stud_r * 0.4
                    hole_bbox = [cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r]
                    draw.ellipse(hole_bbox, fill=(100, 100, 100))

    return overlay_img, color_counter, w, h

# -------------------------------------------------
# Streamlit App
# -------------------------------------------------
def main():
    st.title("🧱 LEGO Mosaic Generator")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    st.sidebar.header("Settings")
    studs_w = st.sidebar.number_input("Width (studs)", 20, 500, 100)
    studs_h = st.sidebar.number_input("Height (studs)", 20, 500, 100)
    quality = st.sidebar.selectbox("Quality", ["low", "medium", "high"])
    stud_type = st.sidebar.selectbox("Stud Type", ["round", "flat", "technic"])
    block_size = st.sidebar.slider("Block size (pixels per stud)", 10, 40, 20)
    stud_ratio = st.sidebar.slider("Stud ratio", 0.5, 1.0, 0.75)

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((studs_w, studs_h), Image.LANCZOS)

        # quality map (just reduces #colors)
        quality_map = {"low": 3, "medium": 12, "high": len(lego_palette)}
        n_colors = quality_map[quality]

        lego_img = quantize_image_to_lego(img_resized, lego_palette)

        overlay_img, color_counter, w, h = add_studs_overlay(
            lego_img,
            block_size=block_size,
            stud_ratio=stud_ratio,
            stud_type=stud_type
        )

        st.image(overlay_img, caption="LEGO Mosaic", use_column_width=True)

        # studs per color
        st.subheader("Studs Required per Color")
        for color, count in color_counter.items():
            st.write(f"Color {color}: {count} studs")

        # real-life size
        st.subheader("Real-life Dimensions")
        width_mm = w * STUD_SIZE_MM
        height_mm = h * STUD_SIZE_MM
        width_m = width_mm / 1000
        height_m = height_mm / 1000
        width_in = width_mm / 25.4
        height_in = height_mm / 25.4
        st.write(f"{width_m:.2f}m x {height_m:.2f}m ({width_in:.1f}in x {height_in:.1f}in)")

        # download button
        st.download_button(
            "Download Mosaic Image",
            data=overlay_img.tobytes(),
            file_name="lego_mosaic.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
