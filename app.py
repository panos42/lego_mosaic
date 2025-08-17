import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
from lego_mosaic import quantize_image_to_lego, lego_palette, count_studs, STUD_SIZE_MM

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




# ----------------------------------------------------

from PIL import Image, ImageDraw
import numpy as np
from collections import Counter

# Standard LEGO stud size (8mm)
STUD_WIDTH_MM = 8

def add_studs_overlay(img, block_size=20, stud_ratio=0.75, stud_type="round"):
    """
    Add LEGO stud visuals on top of an image to simulate actual LEGO bricks.

    Parameters:
        img (PIL.Image): The quantized LEGO mosaic image.
        block_size (int): Size of each LEGO block in pixels for visualization.
        stud_ratio (float): Diameter ratio of stud relative to block size (0 < stud_ratio <= 1).
        stud_type (str): Type of stud: "round" (classic), "flat" (tile), "technic" (with hole).

    Returns:
        PIL.Image: Image with studs overlay.
    """
    arr = np.array(img)
    h, w, _ = arr.shape

    # Create new scaled-up image
    overlay_img = Image.new("RGB", (w*block_size, h*block_size), (255, 255, 255))
    draw = ImageDraw.Draw(overlay_img)

    # Count studs per color
    color_counter = Counter()

    for y in range(h):
        for x in range(w):
            color = tuple(arr[y, x])
            color_counter[color] += 1

            # Draw block rectangle
            x0, y0 = x*block_size, y*block_size
            x1, y1 = x0 + block_size, y0 + block_size
            draw.rectangle([x0, y0, x1, y1], fill=color)

            # Draw stud depending on type
            if stud_type != "flat":
                stud_d = block_size * stud_ratio
                stud_r = stud_d / 2
                cx, cy = x0 + block_size/2, y0 + block_size/2
                bbox = [cx - stud_r, cy - stud_r, cx + stud_r, cy + stud_r]

                if stud_type == "round":
                    highlight = tuple(min(255, int(c*1.1)) for c in color)
                    draw.ellipse(bbox, fill=highlight, outline=(50,50,50))
                elif stud_type == "technic":
                    draw.ellipse(bbox, fill=color, outline=(50,50,50))
                    hole_r = stud_r * 0.4
                    hole_bbox = [cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r]
                    draw.ellipse(hole_bbox, fill=(100,100,100))

    return overlay_img, color_counter, w, h

def print_dimensions_and_examples(w, h):
    width_mm = w * STUD_WIDTH_MM
    height_mm = h * STUD_WIDTH_MM
    width_m = width_mm / 1000
    height_m = height_mm / 1000
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4

    print(f"\nReal-life dimensions:")
    print(f"  mm: {width_mm} x {height_mm}")
    print(f"  meters: {width_m:.2f} x {height_m:.2f}")
    print(f"  inches: {width_in:.2f} x {height_in:.2f}")

    print("\nExample popular sizes:")
    print("  LEGO Art set (e.g., The Beatles) ~ 53cm x 53cm")
    print("  Standard poster ~ 61cm x 91cm (24in x 36in)")
    print("  LEGO Mosaic small display ~ 30cm x 30cm")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LEGO Mosaic with Stud Overlay")
    parser.add_argument("--input", required=True, help="Path to input LEGO mosaic image")
    parser.add_argument("--output", default="lego_final.png", help="Output file name")
    parser.add_argument("--block_size", type=int, default=20, help="Size of each LEGO block in pixels")
    parser.add_argument("--stud_ratio", type=float, default=0.75, help="Stud diameter ratio relative to block size")
    parser.add_argument("--stud_type", choices=["round","flat","technic"], default="round", help="Type of stud")
    parser.add_argument("--resize_w", type=int, help="Resize mosaic width in studs")
    parser.add_argument("--resize_h", type=int, help="Resize mosaic height in studs")

    args = parser.parse_args()

    # Load image
    img = Image.open(args.input).convert("RGB")

    # Resize if requested
    if args.resize_w and args.resize_h:
        img = img.resize((args.resize_w, args.resize_h), Image.LANCZOS)

    overlay_img, color_counter, w, h = add_studs_overlay(
        img,
        block_size=args.block_size,
        stud_ratio=args.stud_ratio,
        stud_type=args.stud_type
    )

    # Show and save
    overlay_img.show()
    overlay_img.save(args.output)

    # Print stud/color summary
    print("\nStuds required per color:")
    for color, count in color_counter.items():
        print(f"  Color {color}: {count} studs")

    # Print real-life dimensions
    print_dimensions_and_examples(w, h)



#-------------------------------------------------------

# lego_mosaic.py
import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# LEGO stud size
STUD_SIZE_MM = 8  # 8mm x 8mm per stud

# Define LEGO colors
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

# Helper functions
def load_and_resize(input_path, studs_w, studs_h):
    img = Image.open(input_path).convert("RGB")
    img_resize = img.resize((studs_w, studs_h), Image.LANCZOS)
    return img_resize

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
            arr[y,x] = np.array(nearest_lego_color(arr[y,x], palette), dtype=np.uint8)
    return Image.fromarray(arr)

def count_studs(img, palette):
    arr = np.array(img)
    counts = {color.name: 0 for color in palette}
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            px = tuple(arr[y,x])
            for color in palette:
                if px == color.rgb:
                    counts[color.name] += 1
    return counts

def parse_real_life_size(size_str):
    try:
        w_m, h_m = map(float, size_str.split("*"))
        return w_m, h_m
    except:
        raise ValueError("Invalid format for --real_life_size. Use WIDTH*HEIGHT in meters, e.g., 0.5*1.0")

# Reduce colors using KMeans based on quality
def reduce_colors(img, n_colors):
    arr = np.array(img).reshape(-1,3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    return Image.fromarray(new_colors.reshape(img.size[1], img.size[0], 3).astype(np.uint8))

def main():
    parser = argparse.ArgumentParser(description="Lego Mosaic Generator")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--studs_w", type=int, default=None, help="Width in studs")
    parser.add_argument("--studs_h", type=int, default=None, help="Height in studs")
    parser.add_argument("--real_life_size", type=str, default=None,
                        help="Real life size in meters: WIDTH*HEIGHT, e.g., 0.5*1.0")
    parser.add_argument("--quality", choices=["low","medium","high"], default="high",
                        help="Overall mosaic quality / number of LEGO colors")
    args = parser.parse_args()

    # Determine studs based on real-life size if specified
    if args.real_life_size:
        w_m, h_m = parse_real_life_size(args.real_life_size)
        studs_w = int(round(w_m*1000 / STUD_SIZE_MM))
        studs_h = int(round(h_m*1000 / STUD_SIZE_MM))
    else:
        studs_w = args.studs_w or 48
        studs_h = args.studs_h or 48

    print(f"Target mosaic size: {studs_w}x{studs_h} studs")

    # Map quality to number of colors
    quality_map = {"low": 3, "medium": 12, "high": len(lego_palette)}
    n_colors = quality_map[args.quality]

    # Load & resize
    resized_img = load_and_resize(args.input, studs_w, studs_h)

    # Reduce colors based on quality
    reduced_img = reduce_colors(resized_img, n_colors)

    # Quantize to LEGO palette
    lego_img = quantize_image_to_lego(reduced_img, lego_palette)

    # Count studs
    counts = count_studs(lego_img, lego_palette)
    print("\nStuds count per color:")
    for name, count in counts.items():
        if count > 0:
            print(f"{name}: {count}")

    # Compute real-life dimensions
    width_m = studs_w * STUD_SIZE_MM / 1000
    height_m = studs_h * STUD_SIZE_MM / 1000
    width_in = width_m * 39.3701
    height_in = height_m * 39.3701
    print(f"\nReal-life size: {width_m:.2f}m x {height_m:.2f}m ({width_in:.1f}in x {height_in:.1f}in)")

    # Show & save
    lego_img.show()
    lego_img.save("lego_mosaic.png")
    print("\nSaved mosaic image as 'lego_mosaic.png'")

if __name__ == "__main__":
    main()
