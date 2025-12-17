import streamlit as st
import pandas as pd
import itertools
import io
import re
import math

# Page configuration
st.set_page_config(page_title="退石最優化計算工具", layout="wide")
st.image("https://cdn-icons-png.flaticon.com/512/616/616490.png", width=80)

class StoneOptimizer:
    """Class to handle stone optimization calculations"""
    
    def __init__(self):
        self.col_pcs = "pcs"
        self.col_weight = "cts"
        self.col_ref = "Ref"
    
    @staticmethod
    def safe_float(val) -> float:
        try:
            return float(val) if val else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    @staticmethod
    def valid_3_decimal(val) -> str:
        try:
            if not val:
                return ""
            f = float(val)
            if f < 0:
                return ""
            s = str(f)
            if '.' in s:
                int_part, dec_part = s.split('.')
                return int_part + '.' + dec_part[:3]
            return s
        except (ValueError, TypeError):
            return ""
    
    # 新增 DP 方法取代 Greedy 和窮舉
    def find_dp_combination(self, available_stones: list[float], target_count: int, 
                             target_weight: float, tolerance: float) -> tuple[list[int], float] | None:
        n = len(available_stones)
        scale = 1000  # 離散化精度
        stones_int = [int(w * scale) for w in available_stones]
        target_int = int(target_weight * scale)
        tol_int = int(tolerance * scale)
        S = sum(sorted(stones_int, reverse=True)[:target_count]) + tol_int  # 上限 sum
        
        # DP 表: dp[num][s] = 是否能用 num 顆達到 sum s
        dp = [[False] * (S + 1) for _ in range(target_count + 1)]
        dp[0][0] = True
        
        # 追蹤組合的父指針 (為了重建組合)
        parent = [[-1] * (S + 1) for _ in range(target_count + 1)]
        
        for i in range(n):
            stone = stones_int[i]
            for num in range(target_count, 0, -1):
                for s in range(S, stone - 1, -1):
                    if dp[num - 1][s - stone] and not dp[num][s]:
                        dp[num][s] = True
                        parent[num][s] = i  # 記錄用到的石頭
        
        # 找在 [target - tol, target + tol] 內的 s
        min_diff = float('inf')
        best_s = -1
        for s in range(max(0, target_int - tol_int), min(S + 1, target_int + tol_int + 1)):
            if dp[target_count][s]:
                diff = abs(s - target_int)
                if diff < min_diff:
                    min_diff = diff
                    best_s = s
        
        if best_s == -1:
            return None
        
        # 重建組合
        combo_indices = []
        num = target_count
        s = best_s
        while num > 0:
            idx = parent[num][s]
            if idx == -1:
                break
            combo_indices.append(idx)
            s -= stones_int[idx]
            num -= 1
        
        total_assigned = best_s / scale
        return combo_indices, total_assigned
    
    def calculate_optimal_assignment(self, stones: list[float], package_rules: list[dict], 
                                     tolerance: float, labels: dict[str, str]) -> list[dict]:
        results = []
        used_indices = set()
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_packages = len(package_rules)
        
        for idx, rule in enumerate(package_rules):
            count = int(rule[self.col_pcs])
            target = float(rule[self.col_weight])
            pack_id = rule.get(self.col_ref, "")
            
            progress_text.text(f"正在處理分包 {idx+1}/{total_packages}: {pack_id or f'第{idx+1}包'} (pcs={count})")
            progress_bar.progress((idx + 1) / total_packages)
            
            available_indices = [i for i in range(len(stones)) if i not in used_indices]
            available_weights = [stones[i] for i in available_indices]
            
            # 統一用 DP 計算
            match = self.find_dp_combination(available_weights, count, target, tolerance)
            
            if match:
                local_indices, total_assigned = match
                global_indices = [available_indices[i] for i in local_indices]
                combo_weights = [stones[i] for i in global_indices]
                
                result_row = {
                    labels["assigned_stones"]: combo_weights,
                    labels["assigned_weight"]: f"{total_assigned:.3f}",
                    labels["expected_weight"]: f"{target:.3f}",
                    labels["diff"]: f"{abs(total_assigned - target):.3f}"
                }
                if pack_id:
                    result_row[self.col_ref] = pack_id
                
                results.append(result_row)
                used_indices.update(global_indices)
            else:
                result_row = {
                    labels["assigned_stones"]: labels["no_match"],
                    labels["assigned_weight"]: "-",
                    labels["expected_weight"]: f"{target:.3f}",
                    labels["diff"]: "-"
                }
                if pack_id:
                    result_row[self.col_ref] = pack_id
                result_row.append(result_row)
        
        progress_bar.empty()
        progress_text.empty()
        
        return results

# 以下是原程式其他部分... (保持不變)
# (省略完整代碼，但您可以直接把這個 class 替換進去)
