# mathGemma-2-9b
This model is based on the Gemma-2-9b architecture and has been fine-tuned using two math problem datasets to improve its accuracy in solving mathematical tasks.

## Datasets

1. **[Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)**:  
   A dataset containing approximately 200K grade school math word problems, with answers generated using Azure GPT-4 Turbo.
   Designed to help models solve elementary-level math problems. 
2. **[MathQA](https://math-qa.github.io/)**:  
   An annotated dataset of math word problems derived from the AQuA-RAT dataset using a novel representation language.  
   The dataset includes questions, multiple-choice options, rationales, and correct answers.
   
## Training Details

The training process included:
- Optimizer: AdamW (8-bit)
- Learning Rate: 2e-4
- Epochs: 1 epoch for Orca-Math, 3 epochs for MathQA
- Batch Size: 16
- Compute Resources: The model was fine-tuned using a single GPU (A100 80GB) for 14 hours.
- Fine-tuning Method: LoRA was used for efficient training and parameter reduction.
- Framework: Fine-tuning was conducted using Unsloth, enabling faster training and better memory efficiency.

## Evaluation
The model was evaluated using the **MathQA test dataset(2985 examples)** with **accuracy** as the primary metric. The following table compares its performance to other models:

| Model                | Accuracy (%)  |
|----------------------|---------------|
| Gemma-2-9b (base)    | 24.02       |
| Mistral-7B-Instruct   | 22.61       |
| Llama-3.1-8b-Instruct | 27.37      |
| Llama-3.2-3b-Instruct | 23.48      |
| Qwen2.5-7B-Instruct  | 38.69         |
| **mathGemma-2-9b**  | **48.91**    |


## How to Get Started with the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

local_model_path = "Dasool/mathGemma-2-9b"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please select the correct answer for the following math problem from the options provided.

### Input:
Problem: Calculate the square root of 144.
Options: 
a) 10
b) 11
c) 12
d) 13

### Response:
"""

inputs = tokenizer(alpaca_prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=100,  
    num_beams=5,
    early_stopping=True,
    temperature=0.7,
    no_repeat_ngram_size=2
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Answer: {answer}")
```

## Limitations

The evaluation is based solely on accuracy for a 5-option multiple-choice task. This provides a high-level performance metric but does not fully capture the model's reasoning ability or performance on more complex, open-ended math problems. A deeper analysis is required to explore the model's problem-solving skills.



> If you have any questions or feedback, feel free to contact: dasolcoi@yonsei.ac.kr
