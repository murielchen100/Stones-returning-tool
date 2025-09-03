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
        """Safely convert value to float"""
        try:
            return float(val) if val else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    @staticmethod
    def valid_3_decimal(val) -> str:
        """Validate and format value to 3 decimal places"""
        try:
            if not val:
                return ""
            f = float(val)
            if f < 0:
                return ""  # Prevent negative weights
            s = str(f)
            if '.' in s:
                int_part, dec_part = s.split('.')
                return int_part + '.' + dec_part[:3]
            return s
        except (ValueError, TypeError):
            return ""
    
    @staticmethod
    def truncate_3_decimal(val) -> str:
        """Truncate to 3 decimal places"""
        try:
            f = float(val)
            if f < 0:
                return "0.000"  # Prevent negative weights
            truncated = math.trunc(f * 1000) / 1000
            return f"{truncated:.3f}"
        except (ValueError, TypeError):
            return "0.000"
    
    def find_first_combination(self, available_stones: list[float], target_count: int, 
                            target_weight: float, tolerance: float) -> tuple[list[int], float] | None:
        """Find the first combination of stones that matches the target within tolerance"""
        
        for combo_indices in itertools.combinations(range(len(available_stones)), target_count):
            combo_weights = [available_stones[i] for i in combo_indices]
            total_weight = sum(combo_weights)
            diff = abs(total_weight - target_weight)
            
            if diff <= tolerance:
                return (list(combo_indices), total_weight)
        
        return None
    
    def calculate_optimal_assignment(self, stones: list[float], package_rules: list[dict], 
                                   tolerance: float, labels: dict[str, str]) -> list[dict]:
        """Calculate optimal stone assignment"""
        results = []
        used_indices = set()
        
        for rule in package_rules:
            count = int(rule[self.col_pcs])
            target = float(rule[self.col_weight])
            pack_id = rule.get(self.col_ref, "")
            
            # Get available stone indices
            available_indices = [i for i in range(len(stones)) if i not in used_indices]
            available_weights = [stones[i] for i in available_indices]
            
            # Find first combination
            first_match = self.find_first_combination(available_weights, count, target, tolerance)
            
            if first_match:
                local_indices, total_assigned = first_match
                # Convert local indices to global indices
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
                # No match found
                result_row = {
                    labels["assigned_stones"]: labels["no_match"],
                    labels["assigned_weight"]: "-",
                    labels["expected_weight"]: f"{target:.3f}",
                    labels["diff"]: "-"
                }
                
                if pack_id:
                    result_row[self.col_ref] = pack_id
                
                results.append(result_row)
        
        return results

def get_language_labels(lang: str) -> dict[str, str]:
    """Get language-specific labels"""
    if lang == "中文":
        return {
            "header": "💎 退石最優化計算工具",
            "mode_label": "選擇輸入方式",
            "upload_label": "上傳 Excel 檔案",
            "package_label": "上傳分包資訊 Excel",
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
            "clear_all": "清除全部"
        }
    else:
        return {
            "header": "💎 Stones Returning Optimizer",
            "mode_label": "Select input mode",
            "upload_label": "Upload Excel file",
            "package_label": "Upload packs info Excel",
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
            "clear_all": "Clear all"
        }

def create_stone_input_grid(labels: dict[str, str]) -> list[float]:
    """Create stone weight input grid"""
    st.subheader(labels["stones_label"])
    st.markdown(f'<span style="font-size:14px; color:gray;">單位：{labels["cts"]}</span>', 
                unsafe_allow_html=True)
    
    if st.button(labels["clear_all"], key="clear_stones"):
        for idx in range(30):
            st.session_state[f"stone_{idx}"] = ""
        st.rerun()
    
    stone_weights = []
    for row in range(6):  # 6 rows x 5 cols = 30 inputs
        cols = st.columns(5)
        for col in range(5):
            idx = row * 5 + col
            with cols[col]:
                st.markdown(f"**{idx+1}.**")
                raw_val = st.text_input(
                    "", 
                    key=f"stone_{idx}", 
                    label_visibility="collapsed", 
                    max_chars=10, 
                    placeholder="0.000"
                )
                val = StoneOptimizer.valid_3_decimal(raw_val)
                if raw_val and not val:  # Invalid input detected
                    st.warning(labels["invalid_input"], icon="⚠️")
                stone_weights.append(StoneOptimizer.safe_float(val))
    
    return stone_weights

def create_package_rules_input(labels: dict[str, str]) -> list[dict]:
    """Create package rules input section"""
    st.subheader(labels["rule_label"])
    
    if st.button(labels["clear_all"], key="clear_rules"):
        for i in range(10):
            st.session_state[f"pcs_{i}"] = ""
            st.session_state[f"weight_{i}"] = ""
            st.session_state[f"packid_{i}"] = ""
        st.rerun()
    
    # Header
    rule_header = st.columns([0.7, 1.5, 1.5, 2])
    with rule_header[0]:
        st.markdown("**#**")
    with rule_header[1]:
        st.markdown("**pcs**")
    with rule_header[2]:
        st.markdown("**cts**")
    with rule_header[3]:
        st.markdown("**Ref**")
    
    package_rules = []
    for i in range(10):
        cols_rule = st.columns([0.7, 1.5, 1.5, 2])
        
        with cols_rule[0]:
            st.markdown(f"**{i+1}**")
        
        with cols_rule[1]:
            pcs_raw = st.text_input(
                "", 
                key=f"pcs_{i}", 
                label_visibility="collapsed", 
                max_chars=3, 
                placeholder="1"
            )
            pcs_val = re.sub(r"\D", "", pcs_raw)[:3] if pcs_raw else ""
            pcs = int(pcs_val) if pcs_val.isdigit() and int(pcs_val) > 0 else 0
            if pcs_raw and pcs == 0:
                st.warning(labels["invalid_input"], icon="⚠️")
        
        with cols_rule[2]:
            weight_raw = st.text_input(
                "", 
                key=f"weight_{i}", 
                label_visibility="collapsed", 
                max_chars=10, 
                placeholder="0.000"
            )
            weight_val = StoneOptimizer.valid_3_decimal(weight_raw)
            total_weight = StoneOptimizer.safe_float(weight_val)
            if weight_raw and not weight_val:
                st.warning(labels["invalid_input"], icon="⚠️")
        
        with cols_rule[3]:
            pack_id = st.text_input(
                "", 
                key=f"packid_{i}", 
                label_visibility="collapsed", 
                max_chars=20, 
                placeholder="Optional"
            )
        
        if pcs > 0 and total_weight > 0:  # Only add non-empty rules
            rule_dict = {
                "pcs": pcs,
                "cts": total_weight
            }
            if pack_id.strip():
                rule_dict["Ref"] = pack_id.strip()
            package_rules.append(rule_dict)
    
    return package_rules

def main():
    """Main application function"""
    # Language selection
    lang = st.selectbox("選擇語言 / Language", ["中文", "English"])
    labels = get_language_labels(lang)
    
    # Header
    st.header(labels["header"])
    st.markdown('<div style="font-size:18px; color:green; margin-bottom:10px;">by Muriel</div>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mode selection
    mode = st.radio(labels["mode_label"], [labels["upload_label"], labels["keyin_label"]])
    
    optimizer = StoneOptimizer()
    results = []
    
    if mode == labels["keyin_label"]:
        # Manual input mode
        
        # Stone weights input
        stone_weights = create_stone_input_grid(labels)
        
        st.markdown("---")
        
        # Package rules input
        package_rules = create_package_rules_input(labels)
        
        st.markdown("---")
        
        # Tolerance input
        tolerance_raw = st.text_input(
            f"{labels['tolerance']}", 
            value="0.003", 
            key="tolerance_manual", 
            placeholder="0.003"
        )
        tolerance_val = StoneOptimizer.valid_3_decimal(tolerance_raw)
        if tolerance_raw and not tolerance_val:
            st.warning(labels["invalid_input"], icon="⚠️")
        tolerance = StoneOptimizer.safe_float(tolerance_val) or 0.003
        
        st.markdown(f'<div style="text-align:right; color:gray; font-size:14px;">{labels["cts"]}</div>', 
                    unsafe_allow_html=True)
        
        # Check for valid input before calculation
        if not any(w > 0 for w in stone_weights) or not package_rules:
            st.warning(labels["no_data"], icon="⚠️")
        else:
            results = optimizer.calculate_optimal_assignment(
                [w for w in stone_weights if w > 0],  # Filter out zero weights
                package_rules, 
                tolerance, 
                labels
            )
    
    elif mode == labels["upload_label"]:
        # File upload mode
        combined_label = "上傳 Excel 檔案" if lang == "中文" else "Upload Excel file"
        combined_file = st.file_uploader(combined_label, type=["xlsx"], key="combined")
        
        st.markdown("---")
        
        # Tolerance input
        tolerance_raw = st.text_input(
            f"{labels['tolerance']}", 
            value="0.003", 
            key="tolerance_upload", 
            placeholder="0.003"
        )
        tolerance_val = StoneOptimizer.valid_3_decimal(tolerance_raw)
        if tolerance_raw and not tolerance_val:
            st.warning(labels["invalid_input"], icon="⚠️")
        tolerance = StoneOptimizer.safe_float(tolerance_val) or 0.003
        
        st.markdown(f'<div style="text-align:right; color:gray; font-size:14px;">{labels["cts"]}</div>', 
                    unsafe_allow_html=True)
        
        if combined_file:
            try:
                # Load Excel file
                df = pd.read_excel(combined_file)
                df.columns = df.columns.str.lower()
                
                # Validate required columns for packages
                required_cols = ["pcs", "cts"]
                if not all(col in df.columns for col in required_cols):
                    st.error(f"{labels['error_label']}: Missing required columns {required_cols}")
                    st.stop()
                
                # Validate use cts for stones
                if "use cts" not in df.columns:
                    st.error(f"{labels['error_label']}: Missing 'use cts' column for stones")
                    st.stop()
                
                # Extract stones from all rows where use cts is not null
                stones = [StoneOptimizer.safe_float(row.get("use cts")) for _, row in df.iterrows() if pd.notnull(row.get("use cts")) and StoneOptimizer.safe_float(row.get("use cts")) > 0]
                
                # Extract package rules
                package_rules = []
                for _, row in df.iterrows():
                    pcs = row.get("pcs")
                    target_cts = row.get("cts")
                    if pd.notnull(pcs) and pd.notnull(target_cts):
                        pcs = StoneOptimizer.safe_float(pcs)
                        target_cts = StoneOptimizer.safe_float(target_cts)
                        if pcs > 0 and target_cts > 0:
                            rule_dict = {
                                "pcs": int(pcs),
                                "cts": target_cts
                            }
                            if "ref" in df.columns and pd.notnull(row["ref"]) and str(row["ref"]).strip():
                                rule_dict["Ref"] = str(row["ref"]).strip()
                            package_rules.append(rule_dict)
                
                # Check for valid input
                if not stones or not package_rules:
                    st.warning(labels["no_data"], icon="⚠️")
                else:
                    results = optimizer.calculate_optimal_assignment(stones, package_rules, tolerance, labels)
                
            except Exception as e:
                st.error(f"{labels['error_label']}: {str(e)}")
                st.stop()
        else:
            st.info(labels["info_label"])
    
    # Display results
    if results:
        st.markdown("---")
        st.subheader(labels["result_label"])
        
        # Create DataFrame and display with custom formatting
        df = pd.DataFrame(results)
        # Ensure consistent column order with Ref first
        columns = [optimizer.col_ref, labels["assigned_stones"], labels["assigned_weight"], 
                   labels["expected_weight"], labels["diff"]]
        # Filter out columns that don't exist (e.g., if Ref is not present)
        columns = [col for col in columns if col in df.columns]
        df = df[columns]
        
        # Apply formatting for better display
        def format_dataframe(df):
            formatted_df = df.copy()
            # Convert lists in assigned_stones to comma-separated strings
            if labels["assigned_stones"] in formatted_df.columns:
                formatted_df[labels["assigned_stones"]] = formatted_df[labels["assigned_stones"]].apply(
                    lambda x: ", ".join(f"{v:.3f}" for v in x) if isinstance(x, list) else x
                )
            for col in [labels["assigned_weight"], labels["expected_weight"], labels["diff"]]:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: f"{float(x):.3f}" if x != "-" else x
                    )
            return formatted_df
        
        try:
            # Display DataFrame without index
            st.dataframe(format_dataframe(df), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}. Please ensure the input data is valid.")
            st.stop()
        
        # Download functionality
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            format_dataframe(df).to_excel(writer, index=False, sheet_name='Results')
        buffer.seek(0)
        
        st.download_button(
            label=labels["download_label"],
            data=buffer,
            file_name="stone_optimization_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
       
if __name__ == "__main__":
    main()
