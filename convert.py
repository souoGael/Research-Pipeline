from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageFont
import os, random

# Helper: add random strokes to an image
def add_random_strokes(image, stroke_count=5):
    """
    Adds random scribble-like strokes to an image.
    stroke_count controls how many strokes are added.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for _ in range(stroke_count):
        # Random start and end points
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)

        # Random stroke width
        stroke_width = random.randint(1, 4)

        # Draw the stroke
        draw.line([(x1, y1), (x2, y2)], fill="black", width=stroke_width)

    return image


# Add noise generation: different level of scribbles when creating final image and save
def pdf_images(pdf_path: str, output_dir: str, stroke_quantity: int = 0):
    """
    Converts PDF lines to images with optional random strokes.
    stroke_quantity: number of random strokes to add per image (default=0 = none).
    """
    # === Step 1: Extract lines from PDF ===
    reader = PdfReader(pdf_path)
    lines = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            page_lines = text.splitlines()
            lines.extend(page_lines)

    # === Step 2: Convert each line to an image ===
    os.makedirs(output_dir, exist_ok=True)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    for i, line in enumerate(lines):
        # Dummy image to calculate text size
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Create final image
        image = Image.new("RGB", (width + 20, height + 20), "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), line, fill="black", font=font)

        # Add scribbles if requested
        if stroke_quantity > 0:
            image = add_random_strokes(image, stroke_quantity)

        # Save image
        image.save(os.path.join(output_dir, f"{i+1}.png"))


# usage
pdf_path = "Documents/1/1.pdf"
output_dir = "Documents/1/images"
pdf_images(pdf_path, output_dir, stroke_quantity=5) 
print("Conversion successful")