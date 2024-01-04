import argparse
from typing import Optional
from datasets import load_dataset
import os.path as osp
import os


def generate_dataset(output_folder: str, num_samples: Optional[int] = None) -> None:
    """
    Generate a dataset by loading data from Wikipedia and splitting it into train, test, and validation sets.

    Args:
        output_folder (str): The path to the output folder where the generated datasets will be saved.
        num_samples (Optional[int]): The number of samples to include in the dataset. If None, all samples will be included.

    Returns:
        None
    """

    dataset = load_dataset(
        "wikipedia",
        "20220301.simple",
        beam_runner="DirectRunner",
        use_auth_token=False,
    )

    df = dataset.to_pandas()
    if num_samples:
        df = df.iloc[:num_samples]

    # Split the dataset into train, test, and validation sets
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index).sample(frac=0.5, random_state=42)
    validation_df = df.drop(train_df.index).drop(test_df.index)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the datasets to CSV files
    train_df.to_csv(osp.join(output_folder, "train_dataset.csv"), index=False)
    test_df.to_csv(osp.join(output_folder, "test_dataset.csv"), index=False)
    validation_df.to_csv(osp.join(output_folder, "validation_dataset.csv"), index=False)

    print("Datasets generated successfully!")


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
