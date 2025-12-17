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
        
        # Step 1: Greedy 初始解（从小到大）
        indexed = sorted(enumerate(available_stones), key=lambda x: x[1])
        selected = [idx for idx, _ in indexed[:target_count]]
        current_total = sum(available_stones[i] for i in selected)
        current_diff = abs(current_total - target_weight)
        
        # 如果初始解就符合，直接返回
        if current_diff <= tolerance:
            return selected, current_total
        
        best_selected = selected.copy()
        best_total = current_total
        best_diff = current_diff
        
        # Step 2: 局部搜尋 - 關鍵修改：接受任何 ≤ tolerance 的解，並立即返回
        iterations = 0
        max_iterations = 200
        
        for _ in range(max_iterations):
            improved = False
            
            for i in range(target_count):
                in_idx = best_selected[i]
                in_weight = available_stones[in_idx]
                
                remaining = [j for j in range(n) if j not in best_selected]
                
                for out_idx in remaining:
                    out_weight = available_stones[out_idx]
                    new_total = best_total - in_weight + out_weight
                    new_diff = abs(new_total - target_weight)
                    
                    # 關鍵：如果新誤差 ≤ tolerance，立即接受並返回（包括正好 = tolerance）
                    if new_diff <= tolerance:
                        new_selected = best_selected.copy()
                        new_selected[i] = out_idx
                        return new_selected, new_total
                    
                    # 如果沒符合但比目前更好，繼續更新（防止卡在局部）
                    if new_diff < best_diff:
                        best_selected[i] = out_idx
                        best_total = new_total
                        best_diff = new_diff
                        improved = True
            
            if not improved:
                break
        
        # 最後檢查 best 是否符合
        if best_diff <= tolerance:
            return best_selected, best_total
        
        return None
    
    def calculate_optimal_assignment(self, stones: list[float], package_rules: list[dict], 
                                     tolerance: float, labels: dict[str, str], 
                                     use_greedy: bool = False) -> tuple[list[dict], list[float]]:
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
        
        remaining_stones = [stones[i] for i in range(len(stones)) if i not in used_indices]
        remaining_stones.sort()
        
        return results, remaining_stones

# 其餘函數（labels、輸入、main）與上一版完全相同，只貼出關鍵差異部分

# ... (get_language_labels, create_stone_input_grid, create_package_rules_input 與上一版相同)

def main():
    # ... (前半部分相同)
    
    if results:
        st.markdown("---")
        st.subheader(labels["result_label"])
        
        df_result = pd.DataFrame(results)
        columns = [optimizer.col_ref, labels["assigned_stones"], labels["assigned_weight"], 
                   labels["expected_weight"], labels["diff"]]
        columns = [col for col in columns if col in df_result.columns]
        df_result = df_result[columns]
        
        def format_dataframe(df):
            formatted = df.copy()
            if labels["assigned_stones"] in formatted.columns:
                formatted[labels["assigned_stones"]] = formatted[labels["assigned_stones"]].apply(
                    lambda x: ", ".join(f"{v:.3f}" for v in x) if isinstance(x, list) else x
                )
            for col in [labels["assigned_weight"], labels["expected_weight"], labels["diff"]]:
                if col in formatted.columns:
                    formatted[col] = formatted[col].apply(lambda x: f"{float(x):.3f}" if x != "-" else x)
            return formatted
        
        st.dataframe(format_dataframe(df_result), use_container_width=True, hide_index=True)
        
        # 統計資訊
        st.markdown("---")
        st.subheader("分配統計")
        
        total_stones = len(stones) if 'stones' in locals() else len([w for w in stone_weights if w > 0])
        allocated_count = total_stones - len(remaining_stones)
        
        st.success(f"**{labels['stats_allocated']}：{allocated_count} 顆**")
        st.info(f"**{labels['stats_remaining']}：{len(remaining_stones)} 顆**")
        
        if remaining_stones:
            remaining_str = ", ".join(f"{w:.3f}" for w in remaining_stones)
            st.caption(f"{labels['stats_remaining_list']}：{remaining_str}")
        else:
            st.caption("所有石頭皆已成功分配！🎉")
        
        # 下載按鈕
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            format_dataframe(df_result).to_excel(writer, index=False, sheet_name='Results')
        buffer.seek(0)
        
        st.download_button(
            label=labels["download_label"],
            data=buffer,
            file_name="stone_optimization_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
