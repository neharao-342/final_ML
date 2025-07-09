
# next_word_predictor.py

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, GPT2LMHeadModel,
                          Trainer, TrainingArguments, pipeline)
import evaluate
import gradio as gr

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example["text"], return_special_tokens_mask=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Group tokens into blocks
block_size = 128
def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_len = (len(concatenated) // block_size) * block_size
    result = {
        "input_ids": [concatenated[i: i + block_size] for i in range(0, total_len, block_size)],
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=1,
    save_strategy="epoch",
    push_to_hub=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# Train
trainer.train()

# Evaluate: Perplexity
perplexity_metric = evaluate.load("perplexity", module_type="metric")
preds = trainer.predict(lm_datasets["validation"]).predictions
results = perplexity_metric.compute(model_id='gpt2', predictions=preds)
print(f"Perplexity: {results['perplexities']}")

# Gradio demo
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
gr.Interface(
    fn=lambda text: pipe(text, max_new_tokens=1)[0]['generated_text'],
    inputs="text", outputs="text", title="Next Word Predictor"
).launch()
