import os
import csv
import random
from tabulate import tabulate
from typing import List, Tuple, Dict, Any

import config # For DATA_DIR, ATTACK_FILES, etc.

def load_single_csv(filepath: str, seen_texts: set) -> List[str]:
    """Loads texts from a single CSV file, ensuring uniqueness across all files."""
    rows = []
    if not os.path.isfile(filepath):
        print(f"⚠️  文件未找到: {filepath}，已跳过。")
        return rows
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header row
            for row in reader:
                if len(row) >= 2: # Assuming text is in the second column (index 1)
                    text = row[1].strip()
                    if text and text not in seen_texts:
                        rows.append(text)
                        seen_texts.add(text)
    except Exception as e:
        print(f"❌ 读取文件 {os.path.basename(filepath)} 时出错: {e}")
    return rows

def load_and_sample_datasets(
    data_dir: str = config.DATA_DIR,
    attack_files: List[str] = config.ATTACK_FILES,
    harmless_files: List[str] = config.HARMLESS_FILES,
    sample_size: int = config.DEFAULT_SAMPLE_SIZE,
    base_test_cases: List[Tuple[str, bool]] = None
) -> List[Tuple[str, bool]]:
    """
    Loads data from specified CSV files, performs stratified sampling,
    and combines with base test cases.
    Returns a list of (text, should_block_label).
    """
    if base_test_cases is None:
        test_cases: List[Tuple[str, bool]] = []
    else:
        test_cases = list(base_test_cases) # Make a mutable copy

    file_specs: List[Dict[str, Any]] = []
    seen_texts_global: set = {text for text, _ in test_cases} # Initialize with texts from base_test_cases

    # Load attack samples
    for fname in attack_files:
        fpath = os.path.join(data_dir, fname)
        rows = load_single_csv(fpath, seen_texts_global)
        if rows:
            file_specs.append({"fname": fname, "label": True, "rows": rows, "type": "Attack"})

    # Load harmless samples
    for fname in harmless_files:
        fpath = os.path.join(data_dir, fname)
        rows = load_single_csv(fpath, seen_texts_global)
        if rows:
            file_specs.append({"fname": fname, "label": False, "rows": rows, "type": "Benign"})

    if not file_specs:
        print("⚠️  未找到任何外部样本文件，仅使用内置测试用例（如果提供）。")
        return test_cases

    # Calculate quotas for sampling
    num_files = len(file_specs)
    if num_files == 0: # Should be caught by above, but defensive
        return test_cases

    # Distribute sample_size proportionally or equally
    # This logic aims for balanced sampling if sample_size is the target for *new* samples
    
    # If sample_size is total including base cases, adjust:
    # current_external_samples = sample_size - len(base_test_cases)
    # target_external_samples = max(0, current_external_samples)
    
    target_external_samples = sample_size # Assume sample_size is for external files

    if target_external_samples <= 0:
        print("抽样数量配置为0或负数，或所有样本已由内置用例提供。")
        return test_cases
        
    base_quota = target_external_samples // num_files
    extra_quota = target_external_samples % num_files

    for idx, fs in enumerate(file_specs):
        fs["quota"] = base_quota + (1 if idx < extra_quota else 0)

    # Adjust quotas if a file has fewer samples than its quota
    deficit = 0
    for fs in file_specs:
        if len(fs["rows"]) < fs["quota"]:
            deficit += fs["quota"] - len(fs["rows"])
            fs["quota"] = len(fs["rows"])

    # Redistribute deficit to files that have more samples
    # This loop ensures we try to meet target_external_samples if possible
    while deficit > 0:
        updated_in_pass = False
        eligible_files_for_extra = [fs for fs in file_specs if len(fs["rows"]) > fs["quota"]]
        if not eligible_files_for_extra:
            break # No files can take more samples

        # Distribute deficit among eligible files
        # A simple way is to give one to each until deficit is consumed or files are maxed out
        for fs in eligible_files_for_extra:
            if deficit == 0: break
            fs["quota"] += 1
            deficit -= 1
            updated_in_pass = True
        if not updated_in_pass: # No changes made, avoid infinite loop
            break


    print("\n--- 数据集抽样计划 ---")
    total_sampled_count = 0
    for fs in file_specs:
        num_to_sample = min(fs["quota"], len(fs["rows"])) # Ensure not to sample more than available
        if num_to_sample > 0:
            picked_samples = random.sample(fs["rows"], num_to_sample)
            test_cases.extend([(text, fs["label"]) for text in picked_samples])
            total_sampled_count += num_to_sample
        print(f"  - 文件: {fs['fname']:<20} ({fs['type']}) | 计划抽样: {fs['quota']} | 实际可用: {len(fs['rows'])} | 最终抽样: {num_to_sample}")
    
    print(f"总共从外部文件抽样: {total_sampled_count} 条")
    print(f"测试用例总数 (包括内置): {len(test_cases)} 条")
    print("--- 抽样完成 ---\n")
    
    random.shuffle(test_cases) # Shuffle all test cases together
    return test_cases


def live_update_table(results_for_current_sample: List[Dict[str, Any]]):
    """
    实时更新表格以显示当前样本在不同检测路径下的结果。
    :param results_for_current_sample: A list of result dicts, one for each path for the *current* sample.
                                       e.g., [regex_result, model_result, full_result]
    """
    headers = ["检测路径", "拦截状态", "耗时(ms)", "触发规则/原因 (部分)"]
    rows = []

    for res in results_for_current_sample:
        status_emoji = "🔴 拦截" if res.get("is_blocked", False) else "🟢 放行"
        time_ms_str = f"{res.get('time_cost', 0)*1000:.1f}"
        
        rules_or_reason = res.get("triggered_rules", [])
        if isinstance(rules_or_reason, list):
            display_rules = "\n".join(rules_or_reason[:2]) if rules_or_reason else "无"
        elif isinstance(rules_or_reason, str): # For single string reasons
            display_rules = rules_or_reason[:100] + "..." if len(rules_or_reason) > 100 else rules_or_reason
        else:
            display_rules = "信息不可用"

        rows.append([
            res.get("path", "未知路径"),
            status_emoji,
            time_ms_str,
            display_rules
        ])

    # Clear previous output (works better in some terminals)
    # os.system('cls' if os.name == 'nt' else 'clear') # This can be disruptive, use with caution or avoid.
    
    print("\n" + "~" * 60)
    print("当前样本检测结果:")
    print(tabulate(rows, headers=headers, tablefmt="grid", maxcolwidths=[None, None, None, 40])) # Limit rule column width
    print("~" * 60 + "\n")