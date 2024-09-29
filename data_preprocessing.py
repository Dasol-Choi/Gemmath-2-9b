alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

def format_orca_data(examples):
    """Preprocess Orca-Math dataset."""
    questions = examples["question"]
    answers = examples["answer"]
    texts = []
    
    for question, answer in zip(questions, answers):
        text = alpaca_prompt.format(
            instruction="Solve the following math problem",
            input=question,
            response=answer
        )
        texts.append(text)
    return {"text": texts}

def format_mathqa_data(example):
    """Preprocess MathQA dataset."""
    problem = example["Problem"]
    options = example["options"]
    correct = example["correct"]
    rationale = example["Rationale"]

    return {
        "text": alpaca_prompt.format(
            instruction="Solve the following math problem and provide a rationale.",
            input=f"Problem: {problem}\nOptions: {options}",
            response=f"Answer: {correct}\nRationale: {rationale}"
        )
    }
