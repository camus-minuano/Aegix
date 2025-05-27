from tabulate import tabulate
from typing import List, Dict, Any

import config

class AcademicEvaluator:
    """学术评估指标计算（基于KDD 2024等常见标准）"""
    def __init__(self):
        self.tp = 0  # True Positives: 正确拦截攻击
        self.tn = 0  # True Negatives: 正确放行安全内容
        self.fp = 0  # False Positives: 误拦截安全内容 (Type I error)
        self.fn = 0  # False Negatives: 漏检攻击 (Type II error)

    def update(self, should_block: bool, actual_block: bool):
        """
        更新混淆矩阵。
        :param should_block: bool, 真实标签，是否应该被拦截 (True for attack, False for benign)
        :param actual_block: bool, 检测结果，是否实际被拦截
        """
        if should_block:  # Positive class (Attack)
            if actual_block:
                self.tp += 1
            else:
                self.fn += 1
        else:  # Negative class (Benign)
            if actual_block:
                self.fp += 1
            else:
                self.tn += 1

    @property
    def precision(self) -> float:
        """精确率 (Precision): TP / (TP + FP)"""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float: # Also known as True Positive Rate (TPR) or Sensitivity
        """召回率 (Recall): TP / (TP + FN)"""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 Score: 2 * (Precision * Recall) / (Precision + Recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """准确率 (Accuracy): (TP + TN) / (TP + TN + FP + FN)"""
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def fpr(self) -> float: # False Positive Rate
        """误报率 (FPR): FP / (FP + TN)"""
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0
    
    @property
    def fnr(self) -> float: # False Negative Rate
        """漏报率 (FNR): FN / (TP + FN)"""
        return self.fn / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def get_report_data(self) -> List[List[str]]:
        return [
            ["True Positives (TP)", str(self.tp)],
            ["True Negatives (TN)", str(self.tn)],
            ["False Positives (FP)", str(self.fp)],
            ["False Negatives (FN)", str(self.fn)],
            ["Accuracy", f"{self.accuracy:.2%}"],
            ["Precision (PPV)", f"{self.precision:.2%}"],
            ["Recall (TPR, Sensitivity)", f"{self.recall:.2%}"],
            ["F1 Score", f"{self.f1_score:.2%}"],
            ["False Positive Rate (FPR)", f"{self.fpr:.2%}"],
            ["False Negative Rate (FNR)", f"{self.fnr:.2%}"],
        ]

    def show_report(self):
        print(f"\n=== {config.REPORT_TITLE} ===")
        headers = ["指标 (Metric)", "值 (Value)"]
        table_data = self.get_report_data()
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("=" * (len(config.REPORT_TITLE) + 6))


class ExperimentRecorder:
    """记录和展示不同检测路径的实验统计数据。"""
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {
            # path_key will be like 'regex_only', 'model_only', 'full_flow'
            # or dynamically from result["path"]
        }

    def add_result(self, path_key: str, is_blocked: bool, time_cost: float):
        """
        添加单次检测结果。
        path_key: 区分检测路径的标识符，例如 'regex_only', 'model_only', 'full_flow_combined'
        """
        if path_key not in self.results:
            self.results[path_key] = {'total': 0, 'blocked': 0, 'time_sum': 0.0, 'time_count': 0}

        self.results[path_key]['total'] += 1
        if is_blocked:
            self.results[path_key]['blocked'] += 1
        self.results[path_key]['time_sum'] += time_cost
        self.results[path_key]['time_count'] += 1


    def show_statistics(self):
        headers = ["检测路径", "测试量 (Total)", "拦截数 (Blocked)", "拦截率 (Block Rate)", "平均耗时 (Avg Time ms)"]
        rows = []

        # Sort by a predefined order or alphabetically if not predefined
        path_order = ["输入 → 正则", "输入 → 小模型", "输入 → 正则(拦截) → 小模型", "输入 → 正则(通过) → 小模型"]
        
        sorted_paths = sorted(self.results.keys(), key=lambda p: path_order.index(p) if p in path_order else float('inf'))


        try:
            for path_key in sorted_paths:
                data = self.results.get(path_key)
                if not data or data['total'] == 0:
                    continue

                block_rate = (data['blocked'] / data['total']) * 100 if data['total'] > 0 else 0
                avg_time_ms = (data['time_sum'] / data['time_count']) * 1000 if data['time_count'] > 0 else 0
                
                rows.append([
                    path_key,
                    data['total'],
                    data['blocked'],
                    f"{block_rate:.1f}%",
                    f"{avg_time_ms:.1f}"
                ])
            
            if not rows:
                print("\n[统计数据]: 未记录任何实验结果。")
                return

            print("\n" + "=" * 70)
            print("实验性能统计 (Performance Statistics)")
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            print("=" * 70)

        except Exception as e:
            print("\n[统计数据输出异常]")
            print(f"当前已记录数据: {self.results}")
            print(f"统计输出时发生错误: {e}")