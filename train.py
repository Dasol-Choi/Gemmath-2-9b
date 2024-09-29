from data_preprocessing import format_orca_data, format_mathqa_data
from finetuning import load_model, fine_tune_model
from datasets import load_dataset

# Load the base Gemma-2-9b model
model, tokenizer = load_model()

# Orca-Math dataset
orca_data = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
orca_data = orca_data.map(format_orca_data, batched=True)

# Fine-tune on Orca-Math for 1 epoch
fine_tune_model(model, tokenizer, orca_data, epochs=1, output_dir="orca_finetuned_outputs")

# MathQA dataset
mathqa_data = load_dataset('json', data_files='Datasets/MathQA/train.json')
mathqa_data = mathqa_data["train"].map(format_mathqa_data)

# Fine-tune on MathQA for 3 epochs
fine_tune_model(model, tokenizer, mathqa_data, epochs=3, output_dir="mathqa_finetuned_outputs")

# Save the final fine-tuned model
model.save_pretrained("models/mathGemma-2-9b")
tokenizer.save_pretrained("models/mathGemma-2-9b")
