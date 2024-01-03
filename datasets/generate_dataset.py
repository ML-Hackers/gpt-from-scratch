import argparse
from typing import Optional
from datasets import load_dataset, Dataset
import pandas as pd
import os.path as osp
import os
from transformers import GPT2TokenizerFast


def generate_dataset(
    num_samples: Optional[int] = None, split: str = "test"
) -> Dataset:

    ds = load_dataset("wikimedia/wikipedia", "20231101.en")
    return ds

def tokenize(element):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    context_length= 2048
    
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentiment dataset")
    parser.add_argument(
        "--output_folder", type=str, default="outputs", help="Output folder path"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples per language"
    )
    args = parser.parse_args()

    split="train"
    train = generate_dataset(split=split)
    tokenized_train = train.map(
        tokenize, batched=True, num_proc=16, remove_columns=train["train"].column_names
    )
    os.makedirs(args.output_folder, exist_ok=True)
    tokenized_train.to_csv(osp.join(args.output_folder, f"{split}_dataset.csv"), index=False)
    print(f"Dataset {split} {len(tokenized_train)} generated successfully!")
