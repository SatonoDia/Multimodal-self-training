import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from PIL import Image
import argparse
from tqdm import tqdm
import copy
from collections import Counter
import re
from pathlib import Path
from mathruler.grader import extract_boxed_content, grade_answer
import datasets
import base64

Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(
    description='Evaluate questions for images using LLM')
parser.add_argument('--model_path',
                    default='/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct',
                    help='Path to model')
parser.add_argument('--data_source',
                    default='zjuwh/self_train_set', 
                    help='data source')
parser.add_argument('--input_file',
                    default=None,
                    help='Path to the image folder')
parser.add_argument('--output_file_path',
                    help='save path of answers',
                    default='/root/autodl-tmp/self_train_verl/filter.json')

def load_image_from_dataset_item(image_field):
    """
    Load a PIL Image from various possible dataset field types:
    - PIL.Image.Image → return as-is
    - bytes (PNG/JPEG data) → open with BytesIO
    - str (base64 string) → open with Image.open
    """
    if isinstance(image_field, Image.Image):
        return image_field
    elif isinstance(image_field, bytes):
        from io import BytesIO
        return Image.open(BytesIO(image_field))
    elif isinstance(image_field, str):
        image_bytes = base64.b64decode(image_field, validate=True)
        return Image.open(BytesIO(image_bytes))
    else:
        raise ValueError(f"Unsupported image field type: {type(image_field)}")
    

def find_voting_answer(answer_group):
    """
    Find quasi-GroundTruth using max voting
    """
    if not answer_group:
        return None
    m = len(answer_group)
    counter = Counter(answer_group)    
    max_count = max(counter.values())
    for str in answer_group:
        if counter[str] == max_count:
            return max_count, str
    return max_count, ""

def calculate_accuracy(results, gt):
    correct = 0
    total = len(results)
    for res in results:
        if grade_answer(res, gt):
            correct += 1
    return correct / total if total > 0 else 0.0

def question_evaluate(data_source, dataset,  llm, sampling_params, tokenizer):
    inputs = []
    final_results = []
    valid_data = []
    # Read questions from the json file
    
    system_prompt = """You are an expert competition-math problem solver."""

    answer_prompt = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    for index, item in enumerate(dataset):
        question = item["problem"]
        ground_truth = item["answer"]
        raw_image = item["images"][0]
        image = load_image_from_dataset_item(raw_image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        user_text = f"{answer_prompt}\nQuestion: {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_text}
            ]}
        ]

        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            add_special_tokens=True
        )

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        })

        valid_data.append({
            "img_data_source": data_source,
            "image_idx": index, 
            "question": question,
            "gt": ground_truth        
        })
            

    print(f"Generating answers for {len(inputs)} images...")
    
    # CHUNK_SIZE logic added here
    CHUNK_SIZE = 2000  # Process in chunks of 2000 to avoid memory issues
    all_outputs = []
    
    for i in range(0, len(inputs), CHUNK_SIZE):
        chunk_inputs = inputs[i:i+CHUNK_SIZE]
        print(f"Processing chunk {i//CHUNK_SIZE + 1}/{(len(inputs) + CHUNK_SIZE - 1)//CHUNK_SIZE} with {len(chunk_inputs)} items...")
        chunk_outputs = llm.generate(chunk_inputs, sampling_params)
        all_outputs.extend(chunk_outputs)
        print(f"Chunk {i//CHUNK_SIZE + 1} processed successfully.")
    
    outputs = all_outputs
    # End of CHUNK_SIZE logic
    
    for index, item in enumerate(valid_data):
        results = [extract_boxed_content(output.text) for output in outputs[index].outputs]
        results = [res for res in results if res]
        acc = calculate_accuracy(results, item["gt"])
        final_results.append({
            "img_data_source": data_source,
            "image_idx": item["image_idx"], 
            "question": item["question"],
            "gt": item["gt"],
            "acc": acc,
            "outputs": results
        })
    return final_results

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)   
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        repetition_penalty=1.1,
        max_tokens=2048,
        stop_token_ids=[tokenizer.eos_token_id],
        n=10,
    )

    # Load model and tokenizer once
    print("Loading model...")
    try:
        llm = LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            tensor_parallel_size=4,
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        exit(1)
    # Check if input folder exists
    if args.input_file is not None:
        dataset = datasets.load_dataset(
            args.input_file,
        )
    else:
        dataset = datasets.load_dataset(
            args.data_source,
        )
    # Process images and generate questions
    try:

        data_res = question_evaluate(args.data_source, dataset['train'], llm, sampling_params, tokenizer)
        
        if not data_res:
            print("No answer were generated!")
            exit(1)
            
        # Save results
        print(f"Saving {len(data_res)} generated answer to {args.output_file_path}")
        output_path = Path(args.output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_res, ensure_ascii=False, indent=4)) 
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        exit(1)