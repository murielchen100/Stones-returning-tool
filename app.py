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
        
        # 初始：从小到大排序（確保小石頭優先，適合多小石拼小總重）
        remaining = list(enumerate(available_stones))
        remaining.sort(key=lambda x: x[1])
        
        selected = []
        current_total = 0.0
        
        for _ in range(target_count):
            if not remaining:
                break
            
            # 計算目前已選石頭的平均重量（若無則用目標平均）
            if selected:
                current_avg = current_total / len(selected)
            else:
                current_avg = target_weight / target_count
            
            # 找與當前平均最接近的石頭
            best_idx = None
            best_diff = float('inf')
            best_weight = 0.0
            
            for idx, weight in remaining:
                diff = abs(weight - current_avg)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
                    best_weight = weight
            
            if best_idx is None:
                break
            
            selected.append(best_idx)
            current_total += best_weight
            remaining = [item for item in remaining if item[0] != best_idx]
        
        current_diff = abs(current_total - target_weight)
        
        if current_diff <= tolerance:
            return selected, current_total
        
        # 局部搜尋優化：接受任何 ≤ tolerance 的解，並立即返回
        best_selected = selected.copy()
        best_total = current_total
        best_diff = current_diff
        
        for _ in range(200):
            improved = False
            for i in range(len(best_selected)):
                in_idx = best_selected[i]
                in_weight = available_stones[in_idx]
                
                remaining_indices = [j for j in range(n) if j not in best_selected]
                
                for out_idx in remaining_indices:
                    out_weight = available_stones[out_idx]
                    new_total = best_total - in_weight + out_weight
                    new_diff = abs(new_total - target_weight)
                    
                    if new_diff <= tolerance:
                        new_selected = best_selected.copy()
                        new_selected[i] = out_idx
                        return new_selected, new_total
                    
                    if new_diff < best_diff:
                        best_selected[i] = out_idx
                        best_total = new_total
                        best_diff = new_diff
                        improved = True
            
            if not improved:
                break
        
        if best_diff <= tolerance:
            return best_selected, best_total
        
        return None
    
    def calculate_optimal_assignment(self, stones: list[float], package_rules: list[dict], 
                                     tolerance: float, labels: dict[str, str], 
                                     use_greedy: bool = False) -> tuple[list[dict], list[float]]:
        results = []
        used_indices = set()
        
        # 新增需求1：從 pcs 最小的分包先分配
        package_rules = sorted(package_rules, key=lambda x: x["pcs"])
        
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

# 以下函數（labels、輸入、main）保持不變，只貼關鍵部分

# ... (get_language_labels, create_stone_input_grid, create_package_rules_input 與上一版完全相同)

def main():
    lang = st.selectbox("選擇語言 / Language", ["中文", "English"])
    labels = get_language_labels(lang)
    
    st.header(labels["header"])
    st.markdown('<div style="font-size:18px; color:green; margin-bottom:10px;">by Muriel</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    mode = st.radio(labels["mode_label"], [labels["upload_label"], labels["keyin_label"]])
    
    optimizer = StoneOptimizer()
    results = []
    remaining_stones = []
    
    if mode == labels["keyin_label"]:
        stone_weights = create_stone_input_grid(labels)
        st.markdown("---")
        package_rules = create_package_rules_input(labels)
        st.markdown("---")
        
        tolerance_raw = st.text_input(f"{labels['tolerance']}", value="0.003", key="tolerance_manual", placeholder="0.003")
        tolerance_val = StoneOptimizer.valid_3_decimal(tolerance_raw)
        if tolerance_raw and not tolerance_val:
            st.warning(labels["invalid_input"], icon="⚠️")
        tolerance = StoneOptimizer.safe_float(tolerance_val) or 0.003
        
        if not any(w > 0 for w in stone_weights) or not package_rules:
            st.warning(labels["no_data"], icon="⚠️")
        else:
            max_pcs = max(rule["pcs"] for rule in package_rules)
            use_greedy = max_pcs > 5
            
            results, remaining_stones = optimizer.calculate_optimal_assignment(
                [w for w in stone_weights if w > 0],
                package_rules,
                tolerance,
                labels,
                use_greedy=use_greedy
            )
    
    elif mode == labels["upload_label"]:
        combined_file = st.file_uploader("上傳 Excel 檔案" if lang == "中文" else "Upload Excel file", type=["xlsx"], key="combined")
        st.markdown("---")
        
        tolerance_raw = st.text_input(f"{labels['tolerance']}", value="0.003", key="tolerance_upload", placeholder="0.003")
        tolerance_val = StoneOptimizer.valid_3_decimal(tolerance_raw)
        if tolerance_raw and not tolerance_val:
            st.warning(labels["invalid_input"], icon="⚠️")
        tolerance = StoneOptimizer.safe_float(tolerance_val) or 0.003
        
        if combined_file:
            try:
                df = pd.read_excel(combined_file)
                df.columns = df.columns.str.lower()
                
                required_cols = ["pcs", "cts"]
                if not all(col in df.columns for col in required_cols):
                    st.error(f"{labels['error_label']}: Missing required columns {required_cols}")
                    st.stop()
                
                if "use cts" not in df.columns:
                    st.error(f"{labels['error_label']}: Missing 'use cts' column")
                    st.stop()
                
                stones = []
                for _, row in df.iterrows():
                    w = row.get("use cts")
                    if pd.notnull(w):
                        w_val = StoneOptimizer.safe_float(w)
                        if w_val > 0:
                            stones.append(w_val)
                
                package_rules = []
                for _, row in df.iterrows():
                    pcs = row.get("pcs")
                    target_cts = row.get("cts")
                    if pd.notnull(pcs) and pd.notnull(target_cts):
                        pcs_val = StoneOptimizer.safe_float(pcs)
                        target_val = StoneOptimizer.safe_float(target_cts)
                        if pcs_val > 0 and target_val > 0:
                            rule_dict = {"pcs": int(pcs_val), "cts": target_val}
                            if "ref" in df.columns and pd.notnull(row["ref"]) and str(row["ref"]).strip():
                                rule_dict["Ref"] = str(row["ref"]).strip()
                            package_rules.append(rule_dict)
                
                if not stones or not package_rules:
                    st.warning(labels["no_data"], icon="⚠️")
                else:
                    max_pcs = max(rule["pcs"] for rule in package_rules)
                    use_greedy = max_pcs > 5
                    
                    results, remaining_stones = optimizer.calculate_optimal_assignment(stones, package_rules, tolerance, labels, use_greedy=use_greedy)
                    
            except Exception as e:
                st.error(f"{labels['error_label']}: {str(e)}")
                st.stop()
        else:
            st.info(labels["info_label"])
    
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
