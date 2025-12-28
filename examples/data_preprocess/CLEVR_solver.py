"""
Preprocess a custom JSON dataset (with image_path, question, answer, score)
into verl-compatible parquet format.
Only samples with 0.3 <= score <= 0.7 are kept.
"""

import argparse
import json
import os
from PIL import Image
from io import BytesIO

import datasets
import pandas as pd



def image_to_bytes(image):
    """Convert PIL Image to PNG bytes."""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image")
    byte_io = BytesIO()
    image.convert("RGB").save(byte_io, format="PNG")
    return byte_io.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to input JSON file.")
    parser.add_argument("--local_save_dir", default="data/clevr/data_solver_train", help="Local directory to save parquet files.")
    args = parser.parse_args()
    data_source = "dddraxxx/spatial_clevr_numbered_2000"
    dataset = datasets.load_dataset(data_source)
    # Resolve paths
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Load JSON data
    with open(args.json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # Filter by score
    filtered_data = [
        item for item in data_list
        if "score" in item and isinstance(item["score"], (int, float)) and 0.3 <= item["score"] <= 0.7
    ]
    print(f"Loaded {len(data_list)} samples, kept {len(filtered_data)} with 0.3 <= score <= 0.7.")

    # Load images and prepare records
    records = []
    for idx, item in enumerate(filtered_data):
        image_idx = item.get("image_idx")
        question = item.get("question", "")
        answer = item.get("answer", "")

        if not all([question, answer]):
            print(f"Skipping incomplete sample at index {idx}")
            continue

        dataset_item = dataset["train"][image_idx]
        image = dataset_item["image"]
        image_bytes = image_to_bytes(image)
        records.append({
            "image_bytes": image_bytes,
            "question": question,
            "answer": answer,
            "original_index": idx,
            "score": item["score"]
        })

    if not records:
        raise ValueError("No valid samples after filtering and image loading.")

    # Convert to Hugging Face Dataset
    df = pd.DataFrame(records)
    dataset = datasets.Dataset.from_pandas(df)

    # Define instruction template (same as original)
    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE single option put in \boxed{}."
    )

    def process_fn(example, idx):
        problem = example["question"]
        prompt = "<image>" + problem + " " + instruction_following
        answer = example["answer"]
        image_bytes = example["image_bytes"]

        return {
            "data_source": data_source,  
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "images": [image_bytes],  # Note: verl expects list of images
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": idx,
                "answer": answer,
                "question": problem,
                "original_score": example["score"],
                "original_index": example["original_index"]
            },
        }

    dataset = dataset.map(
        function=process_fn,
        with_indices=True,
        num_proc=8,
        remove_columns=dataset.column_names  # remove raw columns after processing
    )
    from datasets import Sequence, Image
    dataset = dataset.cast_column("images", Sequence(Image()))
    # Save to parquet
    output_path = os.path.join(local_save_dir, "train.parquet")
    dataset.to_parquet(output_path)
    print(f"Saved {len(dataset)} samples to {output_path}")


if __name__ == "__main__":
    main()