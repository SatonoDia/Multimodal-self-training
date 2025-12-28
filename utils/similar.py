import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_zero_acc_image_idxs(data):
    """从 JSON 数据（假设为 list of dict）中提取 acc == 0.0 的 image_idx 集合"""
    idxs = set()
    for item in data:
        if isinstance(item, dict) and item.get("acc") == 0.0 and "image_idx" in item:
            idx = item["image_idx"]
            # 支持 str/int 类型的 image_idx，统一转为 str 避免类型不一致问题
            idxs.add(str(idx))
    return idxs

def main(file1, file2):
    data1 = load_json(file1)
    data2 = load_json(file2)

    set1 = get_zero_acc_image_idxs(data1)
    set2 = get_zero_acc_image_idxs(data2)

    intersection = set1 & set2
    union = set1 | set2

    print(f"File1 中 acc=0.0 的条目数: {len(set1)}")
    print(f"File2 中 acc=0.0 的条目数: {len(set2)}")
    print(f"共同的 image_idx 数（交集）: {len(intersection)}")

    if union:
        jaccard_ratio = len(intersection) / len(union)
        print(f"相同 image_idx 占并集的比例（Jaccard）: {jaccard_ratio:.4f} ({len(intersection)}/{len(union)})")

        # 其他常见定义（可选）：
        ratio_to_file1 = len(intersection) / len(set1) if set1 else 0
        ratio_to_file2 = len(intersection) / len(set2) if set2 else 0
        print(f"占 file1 中 acc=0.0 条目的比例: {ratio_to_file1:.4f}")
        print(f"占 file2 中 acc=0.0 条目的比例: {ratio_to_file2:.4f}")
    else:
        print("两文件中均无 acc=0.0 的条目，无法计算比例。")

main('openvlthinker_data_bak.json', 'openvlthinker_data.json')