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

Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(
    description='Evaluate geometry math questions for images using LLM')
parser.add_argument('--model_path',
                    default='/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5',
                    help='Path to model')
parser.add_argument('--question_file',
                    default='/root/autodl-tmp/self_train_verl/geo3k_questions.json',
                    help='Path to the question file')
parser.add_argument('--output_file_path',
                    help='save path of answers',
                    default='/root/autodl-tmp/self_train_verl/geo3k_answers.json')

def split_output_text(output_text):
    """
    Extract the final answer from model output using <answer> tags.
    """
    
    # Clean up the output text
    text = output_text.strip()
    
    # Extract content between <answer> and </answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        alphabet_match = re.search(r'[A-Z]', answer)
        if alphabet_match:
            answer = alphabet_match.group()
        else:
            return answer
    else:
        answer = text
    return answer

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
    return max_count, None


def question_evaluate(question_file, llm, sampling_params, tokenizer):
    inputs = []
    final_results = []
    valid_data = []
    # Read questions from the json file
    with open(question_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    system_prompt = """You are an expert competition-math problem setter."""

    answer_prompt = """
        Solve the multiple-choice math question based on the provided image. Let's think step by step.
        First, you must fully perceive the image, extracting any valuable visual information from it (including the sizes of labeled angles, lengths of line segments, and various positional relationships.
        Then, apply appropriate mathematical principles to solve the question, give the correct choice option.

        Format your response as: 
        <answer>[Give the correct answer, only the answer option(A B C D)]</answer>

        Output ONLY your final answer choice alphabet A, B, C or D (no other symbol).
        """

    for index, item in enumerate(tqdm(data, desc="Generating Answers")):
        question = item["question"]
        ground_truth = item["answer"]
        image_path = item["image_path"]

        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
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
            "id": item["id"],
            "img_data_source": item["img_data_source"],
            "image_path": item["image_path"],
            "image": item["image"], 
            "question": question,
            "answer": ground_truth        
        })
            

    print(f"Generating answers for {len(inputs)} images...")
    outputs = llm.generate(inputs, sampling_params)
    for index, item in enumerate(valid_data):
        results = [split_output_text(output.text) for output in outputs[index].outputs]
        results = [res for res in results if res]
        score, answer = find_voting_answer(results)
        score = score / len(results)
        final_results.append({
            "id": item["id"],
            "img_data_source": item["img_data_source"],
            "image_path": item["image_path"],
            "image": item["image"], 
            "question": item["question"],
            "answer": answer,
            "score": score
        })
    return final_results

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)   
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        repetition_penalty=1.1,
        max_tokens=4096,
        stop_token_ids=[tokenizer.eos_token_id],
        n=10,
    )

    # Load model and tokenizer once
    print("Loading model...")
    try:
        llm = LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            tensor_parallel_size=4
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        exit(1)

    # Process images and generate questions
    try:

        data_res = question_evaluate(args.question_file, llm, sampling_params, tokenizer)
        
        if not data_res:
            print("No answer were generated!")
            exit(1)
            
        # Save results
        print(f"Saving {len(data_res)} generated answer to {args.output_file_path}")
        with open(args.output_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_res, ensure_ascii=False, indent=4)) 
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        exit(1)