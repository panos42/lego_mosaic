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
