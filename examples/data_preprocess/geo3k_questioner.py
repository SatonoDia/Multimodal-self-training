# # # Copyright 2024 Bytedance Ltd. and/or its affiliates
# # #
# # # Licensed under the Apache License, Version 2.0 (the "License");
# # # you may not use this file except in compliance with the License.
# # # You may obtain a copy of the License at
# # #
# # #     http://www.apache.org/licenses/LICENSE-2.0
# # #
# # # Unless required by applicable law or agreed to in writing, software
# # # distributed under the License is distributed on an "AS IS" BASIS,
# # # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # # See the License for the specific language governing permissions and
# # # limitations under the License.
# # """
# # Preprocess the Geometry3k dataset for questioner training (R-Zero multimodal extension)
# # This script converts geo3k data to parquet format for training a questioner to generate
# # geometry questions from images.
# # """

# # import argparse
# # import os

# # import datasets

# # from verl.utils.hdfs_io import copy, makedirs

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--local_dir", default=None)
# #     parser.add_argument("--hdfs_dir", default=None)
# #     parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
# #     parser.add_argument(
# #         "--local_save_dir", default="data/geo3k/data_questioner_train", help="The save directory for the preprocessed dataset."
# #     )

# #     args = parser.parse_args()
# #     local_dataset_path = args.local_dataset_path

# #     data_source = "hiyouga/geometry3k"

# #     if local_dataset_path is not None:
# #         dataset = datasets.load_dataset(
# #             local_dataset_path,
# #         )
# #     else:
# #         dataset = datasets.load_dataset(
# #             data_source,
# #         )

# #     train_dataset = dataset["train"]
# #     test_dataset = dataset["test"]

# #     # Questioner instruction: adapted for multimodal JSON format to match generator.py and caller.py
# #     questioner_instruction = (
# #         "<image>\n\n"
# #         "You are an expert competition-math problem setter.\n\n"
# #         "Create an AMC 12 level multiple-choice geometry question based on the image.\n"
# #         "To build high-quality multiple-choice questions, you need to go through the following steps:\n\n"
# #         "1. First, you must fully perceive the image, extracting any valuable visual information from it (including the sizes of labeled angles, lengths of line segments, and various positional relationships).\n"
# #         "Remember to clearly distinguish value and label of angel and side (Sometimes \"1\" represent the label of angle, not the value.)\n"
# #         "And generate a concise and precise visual description of the image.\n\n"
# #         "2. Then, write a initial mathematically question that includes all necessary conditions and have a fixed value as an answer.\n"
# #         "Make sure you can get the answer before you set options.\n\n"
# #         "3. Lastly, convert the initial question into a multiple-choice question with a correct option. The specific process is as follows:\n"
# #         "After you make sure the answer of the initial question, the value of the correct answer should be randomly set to the options \"A\" or \"B\" or \"C\" or \"D\"。\n"
# #         "Then create 3 other options that are definitely wrong.\n\n"
# #         "Your MUST response in this JSON format:\n"
# #         "<answer>\n"
# #         "{\n"
# #         "    \"description\": \"Visual description you extract from the image\",\n"
# #         "    \"question\": \"Write a complete multiple-choice question that states all necessary conditions clearly, followed by exactly 4 answer options A B C D\",\n"
# #         "    \"answer\": \"Give the correct answer, only the answer option(A B C D)\"\n"
# #         "}\n"
# #         "</answer>\n"
# #         "DO NOT output anything else—no explanations, no extra markup."
# #     )

# #     def make_map_fn(split):
# #         def process_fn(example, idx):
# #             # For questioner training, we use the image as input and want to generate questions
# #             # The original problem becomes our "ground truth" question for training
# #             original_problem = example.pop("problem")
# #             original_answer = example.pop("answer")
# #             images = example.pop("images")

# #             data = {
# #                 "data_source": data_source,
# #                 "prompt": [
# #                     {
# #                         "role": "user",
# #                         "content": questioner_instruction,
# #                     }
# #                 ],
# #                 "images": images,  # Input images for questioner
# #                 "ability": "question_generation",
# #                 # For questioner training, we'll use custom reward based on solver's uncertainty
# #                 "reward_model": {"style": "custom"},
# #                 "extra_info": {
# #                     "split": split,
# #                     "index": idx,
# #                     "original_problem": original_problem,  # Reference question
# #                     "original_answer": original_answer,    # Reference answer
# #                     "task_type": "questioner",            # Mark as questioner data
# #                     # R-Zero specific fields for uncertainty reward calculation
# #                     "need_solver_feedback": True,
# #                 },
# #             }
# #             return data

# #         return process_fn

# #     train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
# #     test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

# #     hdfs_dir = args.hdfs_dir
# #     local_save_dir = args.local_dir
# #     if local_save_dir is not None:
# #         print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
# #     else:
# #         local_save_dir = args.local_save_dir

# #     train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
# #     test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

# #     if hdfs_dir is not None:
# #         makedirs(hdfs_dir)
# #         copy(src=local_save_dir, dst=hdfs_dir)


# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# Preprocess the Geometry3k dataset for questioner training
# This script converts geo3k data to parquet format for training a questioner to generate
# geometry questions from images.
# """

# import argparse
# import os

# import datasets

# # from verl.utils.hdfs_io import copy, makedirs

# def copy(src, dst):
#     """Placeholder for copy function"""
#     pass

# def makedirs(path):
#     """Placeholder for makedirs function"""
#     pass

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_dir", default=None)
#     parser.add_argument("--hdfs_dir", default=None)
#     parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
#     parser.add_argument(
#         "--local_save_dir", default="data/geo3k/data_questioner_train", help="The save directory for the preprocessed dataset."
#     )

#     args = parser.parse_args()
#     local_dataset_path = args.local_dataset_path

#     data_source = "hiyouga/geometry3k"

#     if local_dataset_path is not None:
#         dataset = datasets.load_dataset(
#             local_dataset_path,
#         )
#     else:
#         dataset = datasets.load_dataset(
#             data_source,
#         )

#     train_dataset = dataset["train"]
#     test_dataset = dataset["test"]

#     # 构建questioner训练指令：训练模型从几何图像生成多选题，输出纯JSON格式
#     # 这个指令会被用于训练数据，教会模型如何基于图像生成几何问题
#     questioner_instruction = (
#         "<image>\n\n"
#         "You are an expert competition-math problem setter.\n\n"
#         "Create an AMC 12 level multiple-choice geometry question based on the image.\n"
#         "To build high-quality multiple-choice questions, you need to go through the following steps:\n\n"
#         "1. First, you must fully perceive the image, extracting any valuable visual information from it (including the sizes of labeled angles, lengths of line segments, and various positional relationships).\n"
#         "Remember to clearly distinguish value and label of angel and side (Sometimes \"1\" represent the label of angle, not the value.)\n"
#         "And generate a concise and precise visual description of the image.\n\n"
#         "2. Then, write a initial mathematically question that includes all necessary conditions and have a fixed value as an answer.\n"
#         "Make sure you can get the answer before you set options.\n\n"
#         "3. Lastly, convert the initial question into a multiple-choice question with a correct option. The specific process is as follows:\n"
#         "After you make sure the answer of the initial question, the value of the correct answer should be randomly set to the options \"A\" or \"B\" or \"C\" or \"D\"。\n"
#         "Then create 3 other options that are definitely wrong.\n\n"
#         "Your MUST response in this JSON format:\n"
#         "{\n"
#         "    \"description\": \"Visual description you extract from the image\",\n"
#         "    \"question\": \"Write a complete multiple-choice question that states all necessary conditions clearly, followed by exactly 4 answer options A B C D\",\n"
#         "    \"answer\": \"Give the correct answer, only the answer option(A B C D)\"\n"
#         "}\n"
#         "DO NOT output anything else—no explanations, no extra markup."
#     )

#     # 数据转换函数：将geo3k原始数据转换为questioner训练格式
#     def make_map_fn(split):
#         def process_fn(example, idx):
#             # 提取geo3k原始数据的各个字段
#             original_problem = example.pop("problem")  # 原始问题（作为参考）
#             original_answer = example.pop("answer")    # 原始答案（作为参考）
#             images = example.pop("images")             # 几何图像（questioner的输入）

#             # 构建verl训练数据格式：图像作为输入，期望输出为基于图像生成的几何问题
#             data = {
#                 "data_source": data_source,
#                 "prompt": [
#                     {
#                         "role": "user",
#                         "content": questioner_instruction,  # 使用上面定义的训练指令
#                     }
#                 ],
#                 "response": '{"description": "Sample geometry description", "question": "Sample question with options A B C D", "answer": "A"}',  # 示例响应格式
#                 "images": images,                         # 几何图像作为模型输入
#                 "ability": "question_generation",         # 标记任务类型为问题生成
#                 "reward_model": {
#                     "style": "custom",
#                     "ground_truth": original_answer    # 添加ground_truth字段，虽然我们的奖励函数不使用它
#                 },
#                 "extra_info": {
#                     "split": split,
#                     "index": idx,
#                     "original_problem": original_problem,  # 原始问题（用于对比参考）
#                     "original_answer": original_answer,    # 原始答案（用于对比参考）
#                     "task_type": "questioner",            # 标记这是questioner训练数据
#                     "need_solver_feedback": True,         # R-Zero需要solver反馈来计算不确定性奖励
#                 },
#             }
#             return data

#         return process_fn

#     train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
#     test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

#     hdfs_dir = args.hdfs_dir
#     local_save_dir = args.local_dir
#     if local_save_dir is not None:
#         print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
#     else:
#         local_save_dir = args.local_save_dir

#     train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
#     test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

#     if hdfs_dir is not None:
#         makedirs(hdfs_dir)
#         copy(src=local_save_dir, dst=hdfs_dir)




"""
Preprocess the Geometry3k dataset for questioner training
This script converts geo3k data to parquet format for training a questioner to generate
geometry questions from images.
"""

import argparse
import os

import datasets

# from verl.utils.hdfs_io import copy, makedirs

def copy(src, dst):
    """Placeholder for copy function"""
    pass

def makedirs(path):
    """Placeholder for makedirs function"""
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="data/geo3k/data_questioner_train", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "hiyouga/geometry3k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        dataset = datasets.load_dataset(
            data_source,
        )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    questioner_instruction = (
        "<image>\n\n"
        "You are an expert competition-math problem setter.\n\n"
        "Create an AMC 12 level multiple-choice geometry question based on the image.\n"
        "To build high-quality multiple-choice questions, you need to go through the following steps:\n\n"
        "1. First, you must fully perceive the image, extracting any valuable visual information from it (including the sizes of labeled angles, lengths of line segments, and various positional relationships).\n"
        "Remember to clearly distinguish value and label of angel and side (Sometimes \"1\" represent the label of angle, not the value.)\n"
        "And generate a concise and precise visual description of the image.\n\n"
        "2. Then, write a initial mathematically question that includes all necessary conditions and have a fixed value as an answer.\n"
        "Make sure you can get the answer before you set options.\n\n"
        "3. Lastly, convert the initial question into a multiple-choice question with a correct option. The specific process is as follows:\n"
        "After you make sure the answer of the initial question, the value of the correct answer should be randomly set to the options \"A\" or \"B\" or \"C\" or \"D\"。\n"
        "Then create 3 other options that are definitely wrong.\n\n"
        "Your MUST response in this JSON format:\n"
        "{\n"
        "    \"description\": \"Visual description you extract from the image\",\n"
        "    \"question\": \"Write a complete multiple-choice question that states all necessary conditions clearly, followed by exactly 4 answer options A B C D\",\n"
        "    \"answer\": \"Give the correct answer, only the answer option(A B C D)\"\n"
        "}\n"
        "DO NOT output anything else—no explanations, no extra markup."
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            original_problem = example.pop("problem")
            original_answer = example.pop("answer")
            images = example.pop("images")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": questioner_instruction,
                    }
                ],
                "response": '{"description": "Sample geometry description", "question": "Sample question with options A B C D", "answer": "A"}',
                "images": images,
                "ability": "question_generation",
                "reward_model": {
                    "style": "custom",
                    "ground_truth": original_answer
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "original_problem": original_problem,
                    "original_answer": original_answer,
                    "task_type": "questioner",
                    "need_solver_feedback": True,
                },
            }
            return data

        return process_fn

    # 对 train_dataset 的处理保持不变 
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)

    # --- 修改对 test_dataset 的处理 
    if len(test_dataset) > 2:
        small_test_dataset = test_dataset.select(range(2))
    else:
        small_test_dataset = test_dataset
    
    short_test_prompt = "<image>\n\nMinimal prompt."

    def make_dummy_test_fn(example, idx):
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": short_test_prompt}],
            "response": '{"description": "dummy", "question": "dummy", "answer": "A"}',
            "images": example["images"],
            "ability": "question_generation",
            "reward_model": {"style": "custom", "ground_truth": ""},
            "extra_info": {
                "split": "test",
                "index": idx,
                "original_problem": "",
                "original_answer": "",
                "task_type": "questioner",
                "need_solver_feedback": False,
            },
        }

    test_dataset = small_test_dataset.map(function=make_dummy_test_fn, with_indices=True, num_proc=2)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)