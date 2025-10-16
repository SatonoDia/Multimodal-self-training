from typing import Dict, List
import json
import os
import time
import random
import requests


STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp")


def generate_temp_filename(prefix="temp", suffix=".json"):
    temp_dir = f"{STORAGE_PATH}/temp_results"
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{temp_dir}/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=1):
    return [lst]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index, i):
    try:
        response = requests.get(f"http://127.0.0.1:{5000+index}/hello?name={i}", timeout=30)
        if response.status_code == 200:
            print(f"[caller] Solver {index} processed {i}")
            return True
        else:
            print(f"[caller] Solver {index} failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"[caller] Solver {index} error: {e}")
        return False

def generate_results(data):
    # 使用1个solver服务评估问题难度
    random_name = generate_temp_filename(prefix="temp_0", suffix=".json")
    
    with open(random_name, 'w') as f:
        json.dump(data, f, indent=4)

    # 调用solver服务
    fetch(0, random_name)

    # 收集solver服务的评估结果
    result_file = random_name.replace('.json', '_results.json')
    with open(result_file, 'r') as f:
        final_results = json.load(f)
    
    os.remove(result_file)
    
    return final_results

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        # 解析questioner生成的JSON格式输出
        parsed_data = json.loads(solution_str)
        question = parsed_data.get("question", "").strip()
        answer = parsed_data.get("answer", "").strip()
        
        if not question or not answer:
            # 格式错误惩罚
            return -1.0
        
        # 构建solver评估数据
        data = [{"question": question, "answer": answer}]
        
        # 获取solver反馈（不确定性分数）
        final_results = generate_results(data)
        
        if len(final_results) > 0 and final_results[0]['question']:
            # 计算R-Zero不确定性奖励
            solver_score = final_results[0]["score"]
            uncertainty_reward = min(solver_score, 1 - solver_score)
            return uncertainty_reward
        else:
            # 格式错误惩罚
            return -1.0
            
    except json.JSONDecodeError:
        # JSON解析失败，格式错误惩罚
        return -1.0
    except Exception as e:
        print(f"Error in compute_score: {str(e)}")
        return -1.0