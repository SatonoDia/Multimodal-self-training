import re

from mathruler.grader import extract_boxed_content, grade_answer


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info):
    # difficulty_score = extra_info.get("original_score", 0.5)
    # difficulty_reward = 0.5 - difficulty_score
    # return 0.5 * difficulty_reward + acc_reward(solution_str, ground_truth, True)
    return acc_reward(solution_str, ground_truth, True)