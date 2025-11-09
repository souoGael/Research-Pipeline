# TODO: Limit this to generates one page reports

from datasets import load_dataset

# Use correct name as shown on the page
dataset = load_dataset("EdinburghNLP/xsum", revision="convert/parquet")

# Print one example
sample = dataset['test'][0]
doc_text = sample['document']
ref_summary = sample['summary'] 

cleaned_text = doc_text.replace("\\n", "\n")
cleaned_summary = ref_summary.replace("\\n", "\n")
# print(cleaned_text)
print("\n" + cleaned_summary)