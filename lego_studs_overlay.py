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
