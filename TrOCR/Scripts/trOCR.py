"""
Fully Automatic TrOCR Evaluation Pipeline 
----------------------------------------------------
- Downloads N samples from XSum dataset
- Generates temporary PDFs
- Converts to images with adjustable scribble noise
- Runs TrOCR + Pegasus summarization
- Evaluates with ROUGE & BERTScore
- Deletes all temp files after run

Adjust:
    NUM_DOCS          = number of samples (e.g., 20, 30, 50)
    STROKE_QUANTITY   = scribbles per image
"""

import os, random, shutil, tempfile
import pytesseract
from jiwer import wer, cer
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, PegasusTokenizer, PegasusForConditionalGeneration
from evaluate import load
from datasets import load_dataset
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ====================================================
# === CONFIGURATION ===
# ====================================================
NUM_DOCS = 10               # adjust (20, 30, 50, etc.)
STROKE_QUANTITY = 5         # number of scribbles per image

# ====================================================
# === HELPERS: SCRIBBLE + PDF CREATION ===
# ====================================================

def add_random_strokes(image, stroke_count=5):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(stroke_count):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        stroke_width = random.randint(1, 4)
        draw.line([(x1, y1), (x2, y2)], fill="black", width=stroke_width)
    return image

def text_to_pdf(text, pdf_path):
    """Generate a simple PDF from text."""
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 72
    for line in text.split("\n"):
        c.drawString(72, y, line[:1000])  # truncate long lines
        y -= 15
        if y < 72:
            c.showPage()
            y = height - 72
    c.save()

def pdf_to_images(pdf_path, output_dir, stroke_quantity=0):
    """Convert PDF text into per-line images with optional scribbles."""
    reader = PdfReader(pdf_path)
    lines = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            lines.extend(text.splitlines())

    os.makedirs(output_dir, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    for i, line in enumerate(lines):
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), line, font=font)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        image = Image.new("RGB", (width + 20, height + 20), "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), line, fill="black", font=font)
        if stroke_quantity > 0:
            image = add_random_strokes(image, stroke_quantity)
        image.save(os.path.join(output_dir, f"{i+1}.png"))

# ====================================================
# === MODELS ===
# ====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device)

pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

rouge_metric = load("rouge")
bertscore_metric = load("bertscore")

# ====================================================
# === CORE PIPELINE ===
# ====================================================

def run_trocr(images_dir):
    """OCR all images in a directory and return concatenated text."""
    images = sorted(
        [os.path.join(images_dir, f) for f in os.listdir(images_dir)
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    lines = []
    for path in images:
        image = Image.open(path).convert("RGB")
        pixel_values = trocr_processor(image, return_tensors="pt").pixel_values.to(device)
        ids = trocr_model.generate(pixel_values, max_new_tokens=1000)
        text = trocr_processor.batch_decode(ids, skip_special_tokens=True)[0]
        lines.append(text)
    return "\n".join(lines)

def run_pytesseract(images_dir):
    """Run pytesseract OCR on all images and return concatenated text."""
    images = sorted(
        [os.path.join(images_dir, f) for f in os.listdir(images_dir)
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    lines = []
    for path in images:
        image = Image.open(path).convert("RGB")
        text = pytesseract.image_to_string(image)
        lines.append(text.strip())
    return "\n".join(lines)

def compare_ocr_metrics(ground_truth, prediction):
    """Compute CER, WER, and CAR (Character Accuracy Rate)."""
    c_err = cer(ground_truth, prediction)
    w_err = wer(ground_truth, prediction)
    c_acc = 1 - c_err
    return c_err, w_err, c_acc

def summarize_text(text):
    tokens = pegasus_tokenizer(text, truncation=True, padding='longest', return_tensors="pt")
    summary_ids = pegasus_model.generate(**tokens)
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ====================================================
# === MAIN EVALUATION ===
# ====================================================

def main(num_docs=NUM_DOCS, stroke_quantity=STROKE_QUANTITY):
    # Create temp workspace
    temp_root = tempfile.mkdtemp()
    print(f"\nTemporary workspace: {temp_root}")

    dataset = load_dataset("EdinburghNLP/xsum", split="test")

    def is_one_page(text, max_chars=1500, max_lines=45):
        return len(text) <= max_chars and len(text.splitlines()) <= max_lines

    filtered_dataset = [sample for sample in dataset if is_one_page(sample["document"])]

    rouge1_scores, rouge2_scores, rouge_scores, bert_scores = [], [], [], []

    cer_scores, wer_scores, car_scores = [], [], []

    try:
        for i in range(num_docs):
            print(f"\n================ Document {i+1}/{num_docs} ================")
            sample = filtered_dataset[i]
            doc_text, ref_summary = sample["document"], sample["summary"]

            doc_dir = os.path.join(temp_root, f"doc_{i+1}")
            os.makedirs(doc_dir, exist_ok=True)
            pdf_path = os.path.join(doc_dir, "doc.pdf")
            text_to_pdf(doc_text, pdf_path)

            # Convert to images
            # CLEAN images (no scribbles) for pytesseract (ground truth)
            images_clean_dir = os.path.join(doc_dir, "images_clean")
            pdf_to_images(pdf_path, images_clean_dir, stroke_quantity=0)

            # NOISY images for TrOCR (what the model sees)
            images_dir = os.path.join(doc_dir, "images")
            pdf_to_images(pdf_path, images_dir, stroke_quantity)

            # OCR + summarization
            pytess_text = run_pytesseract(images_clean_dir)
            ocr_text = run_trocr(images_dir)

            # Compute and store OCR metrics
            c_err, w_err, c_acc = compare_ocr_metrics(pytess_text, ocr_text)
            cer_scores.append(c_err)
            wer_scores.append(w_err)
            car_scores.append(c_acc)

            print(f"\n--- TrOCR Benchmark vs pytesseract ---")
            print(f"CER: {c_err:.4f} | WER: {w_err:.4f} | CAR: {c_acc:.4f}")

            generated_summary = summarize_text(ocr_text)

            # Evaluate
            rouge = rouge_metric.compute(predictions=[generated_summary],
                                         references=[ref_summary])
            bert = bertscore_metric.compute(predictions=[generated_summary],
                                            references=[ref_summary],
                                            model_type="microsoft/deberta-xlarge-mnli")

            rouge1_scores.append(rouge["rouge1"])
            rouge2_scores.append(rouge["rouge2"])
            rouge_scores.append(rouge["rougeL"])
            bert_f1 = sum(bert["f1"]) / len(bert["f1"])
            bert_scores.append(bert_f1)

            print(f"ROUGE-1: {rouge['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge['rougeL']:.4f}")
            print(f"BERTScore-F1: {bert_f1:.4f}")

        # === Averages ===
        avg_cer = sum(cer_scores) / len(cer_scores)
        avg_wer = sum(wer_scores) / len(wer_scores)
        avg_car = sum(car_scores) / len(car_scores)

        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        avg_bert = sum(bert_scores) / len(bert_scores)

        print("\n================ FINAL AVERAGES ================")
        print(f"Average CER: {avg_cer:.4f}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average CAR: {avg_car:.4f}")
        print("-----------------------------------------------")
        print(f"Average ROUGE-1: {avg_rouge1:.4f}")
        print(f"Average ROUGE-2: {avg_rouge2:.4f}")
        print(f"Average ROUGE-L: {avg_rouge:.4f}")
        print(f"Average BERTScore-F1: {avg_bert:.4f}")

    finally:
        # Clean up workspace
        shutil.rmtree(temp_root)
        print(f"\nCleaned up temp workspace: {temp_root}")

# ====================================================
# === ENTRY POINT ===
# ====================================================
if __name__ == "__main__":
    main(NUM_DOCS, STROKE_QUANTITY)
