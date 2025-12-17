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
    
    def find_exact_combination(self, available_stones: list[float], target_count: int, 
                               target_weight: float, tolerance: float) -> tuple[list[int], float] | None:
        for combo_indices in itertools.combinations(range(len(available_stones)), target_count):
            combo_weights = [available_stones[i] for i in combo_indices]
            total_weight = sum(combo_weights)
            if abs(total_weight - target_weight) <= tolerance:
                return (list(combo_indices), total_weight)
        return None
    
    def find_greedy_with_local_search(self, available_stones: list[float], target_count: int, 
                                      target_weight: float, tolerance: float) -> tuple[list[int], float] | None:
        if target_count == 0:
            return [], 0.0
        
        n = len(available_stones)
        if n < target_count:
            return None
        
        # Step 1: Greedy 初始解（从小到大優先小石頭）
        indexed = sorted(enumerate(available_stones), key=lambda x: x[1])
        selected = [idx for idx, _ in indexed[:target_count]]
        current_weights = [available_stones[i] for i in selected]
        current_total = sum(current_weights)
        current_diff = abs(current_total - target_weight)
        
        if current_diff <= tolerance:
            return selected, current_total
        
        best_selected = selected.copy()
        best_total = current_total
        best_diff = current_diff
        
        # Step 2: 局部搜尋優化 - 嘗試替換 1 顆石頭
        improved = True
        iterations = 0
        max_iterations = 100  # 防止無限循環
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(target_count):  # 對初始組合中的每顆石頭
                in_idx = best_selected[i]
                in_weight = available_stones[in_idx]
                
                # 剩餘石頭（不在組合中的）
                remaining = [j for j in range(n) if j not in best_selected]
                
                for out_idx in remaining:
                    out_weight = available_stones[out_idx]
                    new_total = best_total - in_weight + out_weight
                    new_diff = abs(new_total - target_weight)
                    
                    if new_diff < best_diff and new_diff <= tolerance:
                        # 找到更好解，更新
                        best_selected[i] = out_idx
                        best_total = new_total
                        best_diff = new_diff
                        improved = True
                        break  # 找到就跳出，重新一輪搜尋
                if improved:
                    break
        
        if best_diff <= tolerance:
            return best_selected, best_total
        
        return None
    
    def calculate_optimal_assignment(self, stones: list[float], package_rules: list[dict], 
                                     tolerance: float, labels: dict[str, str], 
                                     use_greedy: bool = False) -> list[dict]:
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
            
            match = None
            if use_greedy or count > 5:
                match = self.find_greedy_with_local_search(available_weights, count, target, tolerance)
            else:
                match = self.find_exact_combination(available_weights, count, target, tolerance)
            
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
                results.append(result_row)
        
        progress_bar.empty()
        progress_text.empty()
        
        return results

# 以下其餘函數（get_language_labels, create_stone_input_grid, create_package_rules_input, main）完全保持不變
# (為了篇幅省略，但您可以從上一版直接複製過來，這裡只改了 class 內的計算邏輯)

# ... (貼上上一版的 get_language_labels 到 main 結束的所有程式碼)

if __name__ == "__main__":
    main()
