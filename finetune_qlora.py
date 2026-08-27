"""QLoRA fine-tuning for Garud AI's base model.

Fine-tunes a small causal LM (default: Qwen2.5-1.5B-Instruct) on a JSONL
instruction dataset using 4-bit quantization + LoRA adapters, so the whole
run fits on a single modest GPU (e.g. a free-tier Colab T4).

Dataset format (one JSON object per line):
    {"instruction": "...", "response": "..."}

Usage:
    py finetune_qlora.py --data my_data.jsonl --output ./garud-lora --epochs 3

After training, load the base model + adapter at inference time:
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=...)
    model = PeftModel.from_pretrained(model, "./garud-lora")
"""

from __future__ import annotations

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_jsonl_dataset(path: str, tokenizer, max_length: int) -> Dataset:
    """Load {"instruction", "response"} pairs and tokenize as chat-formatted text."""
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    def to_text(example: dict) -> dict:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = Dataset.from_list(records).map(to_text)

    def tokenize(example: dict) -> dict:
        return tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length")

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def build_model_and_tokenizer(model_name: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Standard attention/MLP projection targets for Qwen2/Llama-family models.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Garud AI's base model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", required=True, help="Path to a JSONL file of {instruction, response} pairs.")
    parser.add_argument("--output", default="./garud-lora", help="Directory to save the LoRA adapter.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model)
    dataset = load_jsonl_dataset(args.data, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter saved to {args.output}")


if __name__ == "__main__":
    main()
