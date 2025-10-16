# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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



def load_image(image_path):
    """Load image and return as bytes (to be compatible with datasets.Image)."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB to ensure consistency
            img = img.convert("RGB")
            byte_arr = BytesIO()
            img.save(byte_arr, format="PNG")
            return byte_arr.getvalue()
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to input JSON file.")
    parser.add_argument("--local_save_dir", default="~/data/custom_geo", help="Local directory to save parquet files.")
    args = parser.parse_args()
    data_source = "hiyouga/geometry3k"
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
        image_path = item.get("image_path")
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        if not all([image_path, question, answer]):
            print(f"Skipping incomplete sample at index {idx}")
            continue

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping.")
            continue

        image_bytes = load_image(image_path)
        if image_bytes is None:
            continue

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