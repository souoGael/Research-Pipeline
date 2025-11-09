"""
Fully Automatic donut Evaluation Pipeline (Option A)
----------------------------------------------------
- Downloads N samples from XSum dataset
- Generates temporary PDFs
- Converts to images with adjustable blur
- Runs TrOCR + Pegasus summarization
- Evaluates with ROUGE & BERTScore
- Deletes all temp files after run

Adjust:
    NUM_DOCS          = number of samples (e.g., 20, 30, 50)
    RADIUS            = blur intensity per image
"""
import os, random, shutil, tempfile, sys
import pytesseract
from jiwer import wer, cer
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, PegasusTokenizer, PegasusForConditionalGeneration
from evaluate import load
from datasets import load_dataset
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ====================================================
# === CONFIGURATION ===
# ====================================================
NUM_DOCS = 10               # adjust (20, 30, 50, etc.)
RADIUS = 1.2                # blur intensity per image
MAX_NEW_TOKENS = 128        # reduce from 1000 to avoid memory / decoding issues
USE_CPU_FOR_TROCR = False   # set True to force TrOCR to run on CPU (safer on small GPUs)

# ====================================================
# === HELPERS: SCRIBBLE + PDF CREATION ===
# ====================================================
def add_full_blur(image, radius):
    """
    Applies a full-image Gaussian blur with adjustable intensity.
    """
    return image.filter(ImageFilter.GaussianBlur(radius))

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

def pdf_to_images(pdf_path, output_dir, radius=0):
    """Convert PDF text into per-line images with optional blur."""
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
        # measure text
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), line, font=font)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        image = Image.new("RGB", (max(100, width + 20), height + 20), "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), line, fill="black", font=font)
        if radius > 0:
            image = add_full_blur(image, radius)
        image.save(os.path.join(output_dir, f"{i+1}.png"))

# ====================================================
# === MODELS ===
# ====================================================
# device selection for trocr model
default_device = "cuda" if torch.cuda.is_available() else "cpu"
if USE_CPU_FOR_TROCR:
    donut_device = "cpu"
else:
    donut_device = default_device

print(f"Running donut on: {donut_device}")
print(f"Running pipeline default device: {default_device}")

# Load processor & model (TrOCR)
donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# Move model to desired device and set eval mode
donut_model.to(donut_device)
donut_model.eval()

# Ensure important config tokens are set (prevents generate indexing issues)
# The processor has a tokenizer attribute; use its special tokens.
if getattr(donut_processor, "tokenizer", None) is not None:
    if getattr(donut_processor.tokenizer, "cls_token_id", None) is not None:
        donut_model.config.decoder_start_token_id = donut_processor.tokenizer.cls_token_id
    if getattr(donut_processor.tokenizer, "pad_token_id", None) is not None:
        donut_model.config.pad_token_id = donut_processor.tokenizer.pad_token_id

# Load Pegasus (leave on default device; can be moved to GPU if desired)
pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
# Optionally move pegasus_model to GPU if you want speed (commented out by default)
# pegasus_model.to(default_device)

rouge_metric = load("rouge")
bertscore_metric = load("bertscore")

# ====================================================
# === CORE PIPELINE ===
# ====================================================
def run_donut(images_dir):
    """OCR all images in a directory and return concatenated text.

    Robust checks:
      - ensures float32 dtype
      - ensures batch dim
      - wraps generate in no_grad
      - catches and reports generation errors per image
    """
    images = sorted(
        [os.path.join(images_dir, f) for f in os.listdir(images_dir)
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    lines = []
    for path in images:
        image = Image.open(path).convert("RGB")
        proc_out = donut_processor(image, return_tensors="pt")
        pixel_values = proc_out.pixel_values

        # Ensure batch dim and dtype
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        pixel_values = pixel_values.to(dtype=torch.float32, device=donut_device)

        # quick sanity checks
        try:
            pv_min = float(pixel_values.min().cpu())
            pv_max = float(pixel_values.max().cpu())
        except Exception:
            pv_min, pv_max = None, None

        try:
            with torch.no_grad():
                # limit max_new_tokens to something reasonable
                ids = donut_model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
                text = donut_processor.batch_decode(ids, skip_special_tokens=True)[0]
                lines.append(text)
        except RuntimeError as e:
            # Provide helpful debug info instead of crashing the whole run.
            print(f"\nRuntimeError while running donut on file: {path}")
            print("Exception:", e)
            print("pixel_values shape:", tuple(pixel_values.shape))
            print("pixel_values min/max:", pv_min, pv_max)
            # If CUDA issue, suggest a fallback or skip
            if donut_device != "cpu":
                print("Suggestion: Try setting USE_CPU_FOR_donut=True or set CUDA_LAUNCH_BLOCKING=1 for a detailed traceback.")
            print("Skipping this image and continuing.")
            continue
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
    # Pegasus expects strings; tokenize with truncation to avoid super-long inputs
    tokens = pegasus_tokenizer(text, truncation=True, padding='longest', return_tensors="pt")
    # If pegasus_model moved to GPU, ensure tokens are on same device as pegasus_model
    # tokens = {k: v.to(pegasus_model.device) for k, v in tokens.items()}
    summary_ids = pegasus_model.generate(**tokens, max_new_tokens=60)
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ====================================================
# === MAIN EVALUATION ===
# ====================================================
def main(num_docs=NUM_DOCS, radius=RADIUS):
    # Create temp workspace
    temp_root = tempfile.mkdtemp()
    print(f"\nTemporary workspace: {temp_root}")

    dataset = load_dataset("EdinburghNLP/xsum", split="test")

    def is_one_page(text, max_chars=1500, max_lines=45):
        return len(text) <= max_chars and len(text.splitlines()) <= max_lines

    filtered_dataset = [sample for sample in dataset if is_one_page(sample["document"])]
    if len(filtered_dataset) == 0:
        print("No filtered documents found. Exiting.")
        return

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
            pdf_to_images(pdf_path, images_clean_dir, radius=0)

            # NOISY images for TrOCR (what the model sees)
            images_dir = os.path.join(doc_dir, "images")
            pdf_to_images(pdf_path, images_dir, radius)

            # OCR + summarization
            pytess_text = run_pytesseract(images_clean_dir)
            ocr_text = run_donut(images_dir)

            if not ocr_text.strip():
                print("No OCR output (empty). Skipping evaluation for this document.")
                continue

            # Compute and store OCR metrics
            c_err, w_err, c_acc = compare_ocr_metrics(pytess_text, ocr_text)
            cer_scores.append(c_err)
            wer_scores.append(w_err)
            car_scores.append(c_acc)

            print(f"\n--- DONUT Benchmark vs pytesseract ---")
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
        if rouge1_scores:
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
        else:
            print("No successful document evaluations to average.")

    finally:
        # Clean up workspace
        shutil.rmtree(temp_root)
        print(f"\nCleaned up temp workspace: {temp_root}")

# ====================================================
# === ENTRY POINT ===
# ====================================================
if __name__ == "__main__":
    main(NUM_DOCS, RADIUS)
