import argparse
from typing import Optional
from datasets import load_dataset
import pandas as pd
import os.path as osp
import os


def generate_dataset(
    output_folder: str, num_samples: Optional[int] = None, split: str = "test"
) -> None:
    """
    Generate a dataset for sentiment analysis.

    Args:
        output_folder (str): The path to the output folder where the dataset will be saved.
        num_samples (Optional[int], optional): The number of samples to include in the dataset. Defaults to None.
        split (str, optional): The dataset split to generate (e.g., "train", "test"). Defaults to "test".

    Returns:
        None
    """

    dfs = []

    for lang in [
        "arabic",
        "english",
        "french",
        "german",
        "hindi",
        "italian",
        "portuguese",
        "spanish",
    ]:
        dataset = load_dataset(
            "wikipedia", lang, split=split
        )
        df = dataset.to_pandas()
        if num_samples:
            df = df.iloc[:num_samples]
        df["lang"] = lang
        df["label"] -= 1
        dfs.append(df)

    concatenated_df = pd.concat(dfs)

    os.makedirs(output_folder, exist_ok=True)
    concatenated_df.to_csv(osp.join(output_folder, f"{split}_dataset.csv"), index=False)
    print(f"Dataset {split} {len(concatenated_df)} generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentiment dataset")
    parser.add_argument(
        "--output_folder", type=str, default="outputs", help="Output folder path"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples per language"
    )
    args = parser.parse_args()

    generate_dataset(args.output_folder, args.num_samples)
    generate_dataset(args.output_folder, split="train")
    generate_dataset(args.output_folder, split="validation")
