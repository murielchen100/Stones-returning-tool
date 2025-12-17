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
        
        # Greedy 初始解（从小到大）
        indexed = sorted(enumerate(available_stones), key=lambda x: x[1])
        selected = [idx for idx, _ in indexed[:target_count]]
        current_total = sum(available_stones[i] for i in selected)
        current_diff = abs(current_total - target_weight)
        
        if current_diff <= tolerance:
            return selected, current_total
        
        best_selected = selected.copy()
        best_total = current_total
        best_diff = current_diff
        
        # 局部搜尋 - 接受任何 ≤ tolerance 的解，並立即返回
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

def get_language_labels(lang: str) -> dict[str, str]:
    if lang == "中文":
        return {
            "header": "💎 退石最優化計算工具",
            "mode_label": "選擇輸入方式",
            "upload_label": "上傳 Excel 檔案",
            "keyin_label": "直接輸入用石重量",
            "rule_label": "分包資訊 packs info",
            "stones_label": "用石",
            "result_label": "分配結果",
            "download_label": "下載結果 Excel",
            "error_label": "請上傳正確的 Excel 檔案（需包含正確欄位）",
            "info_label": "請上傳檔案或輸入資料以進行計算",
            "no_match": "找不到符合組合",
            "assigned_stones": "分配用石",
            "assigned_weight": "分配重量",
            "expected_weight": "期望重量",
            "diff": "差異值",
            "tolerance": "容許誤差",
            "cts": "cts",
            "invalid_input": "請輸入有效數字（非負數）",
            "no_data": "請至少輸入一個有效用石重量和分包規則",
            "clear_all": "清除全部",
            "stats_allocated": "已成功分配石頭",
            "stats_remaining": "未分配石頭",
            "stats_remaining_list": "未分配石頭重量（由小到大）"
        }
    else:
        return {
            "header": "💎 Stones Returning Optimizer",
            "mode_label": "Select input mode",
            "upload_label": "Upload Excel file",
            "keyin_label": "Key in stones weights",
            "rule_label": "分包資訊 packs info",
            "stones_label": "Stones",
            "result_label": "Result",
            "download_label": "Download result Excel",
            "error_label": "Please upload valid Excel files with correct columns",
            "info_label": "Please upload files or enter data to proceed",
            "no_match": "No match found",
            "assigned_stones": "Assigned stones",
            "assigned_weight": "Assigned Weight",
            "expected_weight": "Expected Weight",
            "diff": "Difference",
            "tolerance": "Tolerance",
            "cts": "cts",
            "invalid_input": "Please enter valid numbers (non-negative)",
            "no_data": "Please provide at least one valid stone weight and package rule",
            "clear_all": "Clear all",
            "stats_allocated": "Successfully allocated stones",
            "stats_remaining": "Unallocated stones",
            "stats_remaining_list": "Unallocated stone weights (sorted ascending)"
        }

def create_stone_input_grid(labels: dict[str, str]) -> list[float]:
    st.subheader(labels["stones_label"])
    st.markdown(f'<span style="font-size:14px; color:gray;">單位：{labels["cts"]}</span>', unsafe_allow_html=True)
    
    if st.button(labels["clear_all"], key="clear_stones"):
        for idx in range(100):
            st.session_state[f"stone_{idx}"] = ""
        st.rerun()
    
    stone_weights = []
    for row in range(20):
        cols = st.columns(5)
        for col in range(5):
            idx = row * 5 + col
            with cols[col]:
                st.markdown(f"**{idx+1}.**")
                raw_val = st.text_input("", key=f"stone_{idx}", label_visibility="collapsed", max_chars=10, placeholder="0.000")
                val = StoneOptimizer.valid_3_decimal(raw_val)
                if raw_val and not val:
                    st.warning(labels["invalid_input"], icon="⚠️")
                stone_weights.append(StoneOptimizer.safe_float(val))
    return stone_weights

def create_package_rules_input(labels: dict[str, str]) -> list[dict]:
    st.subheader(labels["rule_label"])
    
    if st.button(labels["clear_all"], key="clear_rules"):
        for i in range(30):
            st.session_state[f"pcs_{i}"] = ""
            st.session_state[f"weight_{i}"] = ""
            st.session_state[f"packid_{i}"] = ""
        st.rerun()
    
    rule_header = st.columns([0.7, 1.5, 1.5, 2])
    with rule_header[0]: st.markdown("**#**")
    with rule_header[1]: st.markdown("**pcs**")
    with rule_header[2]: st.markdown("**cts**")
    with rule_header[3]: st.markdown("**Ref**")
    
    package_rules = []
    for i in range(30):
        cols_rule = st.columns([0.7, 1.5, 1.5, 2])
        with cols_rule[0]: st.markdown(f"**{i+1}**")
        
        with cols_rule[1]:
            pcs_raw = st.text_input("", key=f"pcs_{i}", label_visibility="collapsed", max_chars=3, placeholder="1")
            pcs_val = re.sub(r"\D", "", pcs_raw)[:3] if pcs_raw else ""
            pcs = int(pcs_val) if pcs_val.isdigit() and int(pcs_val) > 0 else 0
            if pcs_raw and pcs == 0:
                st.warning(labels["invalid_input"], icon="⚠️")
        
        with cols_rule[2]:
            weight_raw = st.text_input("", key=f"weight_{i}", label_visibility="collapsed", max_chars=10, placeholder="0.000")
            weight_val = StoneOptimizer.valid_3_decimal(weight_raw)
            total_weight = StoneOptimizer.safe_float(weight_val)
            if weight_raw and not weight_val:
                st.warning(labels["invalid_input"], icon="⚠️")
        
        with cols_rule[3]:
            pack_id = st.text_input("", key=f"packid_{i}", label_visibility="collapsed", max_chars=20, placeholder="Optional")
        
        if pcs > 0 and total_weight > 0:
            rule_dict = {"pcs": pcs, "cts": total_weight}
            if pack_id.strip():
                rule_dict["Ref"] = pack_id.strip()
            package_rules.append(rule_dict)
    
    return package_rules

def main():
    lang = st.selectbox("選擇語言 / Language", ["中文", "English"])
    labels = get_language_labels(lang)
    
    st.header(labels["header"])
    st.markdown('<div style="font-size:18px; color:green; margin-bottom:10px;">by Muriel</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    mode = st.radio(labels["mode_label"], [labels["upload_label"], labels["keyin_label"]])
    
    optimizer = StoneOptimizer()
    results = []  # 關鍵修正：提前初始化
    remaining_stones = []  # 提前初始化
    
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
    
    # 現在 results 一定存在，不會 NameError
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
