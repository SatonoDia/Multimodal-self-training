import json
import os
import re 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from PIL import Image
import argparse
from tqdm import tqdm
from pathlib import Path

Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(
    description='Generate geometry math questions for images using LLM')
parser.add_argument('--model_path',
                    default='/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5',
                    help='Path to model')
parser.add_argument('--input_file',
                    default='/root/autodl-tmp/data/geo3k',
                    help='Path to the image folder')
parser.add_argument('--output_file_path',
                    help='save path of generated questions',
                    default='/root/autodl-tmp/self_train_verl/geo3k_questions.json')


def split_output_text(output_text):
    """
    Extract the generated question from <question></question> tags.
    """
    text = output_text.strip()
    
    question_match = re.search(r'<question>(.*?)</question>', text, re.DOTALL)
    if question_match:
        question = question_match.group(1).strip()
    else:
        question = None
    
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = None
    
    description_match = re.search(r'<description>(.*?)</description>', text, re.DOTALL)
    if description_match:
        description = description_match.group(1).strip()
    else:
        description = None
    
    return question, answer, description



def process_dataset(input_folder, llm, sampling_params, tokenizer):
    valid_data = []
    inputs = []
    
    # Get all image files from the folder
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    print(f"Found {len(image_files)} images to process")

    user_prompt = '''Create an AMC 12 level multiple-choice geometry question based on the image. Let's think step by step.
            First, you must fully perceive the image, extracting any valuable visual information from it (including the sizes of labeled angles, lengths of line segments, and various positional relationships), 
            and generate a detailed visual description of the image.

            Then, write a mathematically rigorous multiple-choice question that includes ALL necessary conditions. Use phrases like "Given that..." or "If..." to state condition shown in visual description.
            The question must include four options, one of which is the correct answer. Any question type other than multiple-choice is FORBIDDEN. 

            Your MUST response in this format:

            <description>
            [Visual description you extract from the image]
            </description>

            <question>
            [Write a complete multiple-choice question that states all necessary conditions clearly, followed by exactly 4 answer options A B C D]
            </question>
    
            <answer>
            [Give the correct answer, only the answer option(A B C D)]
            </answer>
            DO NOT output anything else—no explanations, no extra markup.
            '''
    
    # Prepare inputs for batch processing
    for image_name in image_files:
        image_path = os.path.join(input_folder, image_name)
        
        try:
            image = Image.open(image_path)
            
            messages = [
                {"role": "system", "content": "You are an expert competition-math problem setter."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}
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
                "id": image_name.split('.')[0],
                "img_data_source": input_folder.split('/')[-1],
                "image_path": image_path,
                "image_name": image_name
            })
            
        except Exception as e:
            print(f"Error processing image {image_name}: {str(e)}")
            continue

    if not inputs:
        print("No valid images found to process!")
        return []

    print(f"Generating questions for {len(inputs)} images...")
    outputs = llm.generate(inputs, sampling_params)
    data_res = []
    
    for idx, item in tqdm(enumerate(valid_data), total=len(valid_data), desc="Storing questions"):
        try:
            output_text = outputs[idx].outputs[0].text.strip()
            question, answer, description = split_output_text(output_text)
          
            result_item = {
                "id": item["id"],
                "img_data_source": item["img_data_source"],
                "image_path": item["image_path"],
                "image": item["image_name"],  # Add image attribute
                "description": description,
                "question": question,         # Add question attribute
                "answer": answer
            }
            
            data_res.append(result_item)
                
        except Exception as e:
            print(f"Error processing output for image {item['image_name']}: {str(e)}")
            continue
        
    return data_res

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = SamplingParams(
        temperature=0.3,              
        repetition_penalty=1.1,  
        max_tokens = 1024,
        stop_token_ids=[tokenizer.eos_token_id],   
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

    # Check if input folder exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input folder {args.input_file} does not exist!")
        exit(1)

    # Process images and generate questions
    try:
        data_res = process_dataset(args.input_file, llm, sampling_params, tokenizer)
        
        if not data_res:
            print("No questions were generated!")
            exit(1)
            
        # Save results
        print(f"Saving {len(data_res)} generated questions to {args.output_file_path}")
        output_path = Path(args.output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_res, ensure_ascii=False, indent=4)) 
            print(f"Successfully generated {len(data_res)} geometry questions!")
                
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        exit(1)