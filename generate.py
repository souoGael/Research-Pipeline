from datasets import load_dataset

# Load dataset
dataset = load_dataset("EdinburghNLP/xsum", revision="convert/parquet")

def is_one_page(text, max_chars=1500, max_lines=45):
    return len(text) <= max_chars and len(text.splitlines()) <= max_lines

# Access the 'train' split (you can also use 'validation' or 'test')
train_data = dataset["test"]

# Filter to keep only short documents (roughly one page)
filtered_dataset = [sample for sample in train_data if is_one_page(sample["document"])]

# Print one example if available
if filtered_dataset:
    sample = filtered_dataset[1]
    doc_text = sample["document"]
    ref_summary = sample["summary"]

    cleaned_text = doc_text.replace("\\n", "\n")
    cleaned_summary = ref_summary.replace("\\n", "\n")

    print(cleaned_text)
    # print("\nSummary:\n", cleaned_summary)
else:
    print("No samples fit the one-page criteria.")
