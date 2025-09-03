import streamlit as st
import pandas as pd
import itertools
import io
import re
import math
from typing import List, Dict, Tuple, Optional

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
            truncated = math.trunc(f * 1000) / 1000
            return f"{truncated:.3f}"
        except (ValueError, TypeError):
            return "0.000"
    
    def find_best_combination(self, available_stones: List[float], target_count: int, 
                            target_weight: float, tolerance: float) -> Optional[Tuple[List[int], float, float]]:
        """Find the best combination of stones that matches the target"""
        best_combo = None
        best_diff = float('inf')
        
        for combo_indices in itertools.combinations(range(len(available_stones)), target_count):
            combo_weights = [available_stones[i] for i in combo_indices]
            total_weight = sum(combo_weights)
            truncated_total = math.trunc(total_weight * 1000) / 1000
            diff = abs(truncated_total - target_weight)
            
            if diff <= tolerance and diff < best_diff:
                best_combo = (list(combo_indices), truncated_total, diff)
                best_diff = diff
        
        return best_combo
    
    def calculate_optimal_assignment(self, stones: List[float], package_rules: List[Dict], 
                                   tolerance: float, labels: Dict[str, str]) -> List[Dict]:
        """Calculate optimal stone assignment"""
        results = []
        available_stones = stones.copy()
        used_indices = set()
        
        # Sort package rules by weight descending for better matching
        sorted_rules = sorted(enumerate(package_rules), 
                            key=lambda x: x[1][self.col_weight], reverse=True)
        
        for original_idx, rule in sorted_rules:
            count = int(rule[self.col_pcs])
            target = float(rule[self.col_weight])
            pack_id = rule.get(self.col_ref, "")
            
            # Get available stone indices
            available_indices = [i for i in range(len(stones)) if i not in used_indices]
            available_weights = [stones[i] for i in available_indices]
            
            # Find best combination
            best_match = self.find_best_combination(available_weights, count, target, tolerance)
            
            if best_match:
                local_indices, total_assigned, diff = best_match
                # Convert local indices to global indices
                global_indices = [available_indices[i] for i in local_indices]
                combo_weights = [stones[i] for i in global_indices]
                
                result_row = {
                    labels["assigned_stones"]: combo_weights,
                    labels["assigned_weight"]: f"{total_assigned:.3f}",
                    labels["expected_weight"]: f"{target:.3f}",
                    labels["diff"]: f"{diff:.3f}"
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

def get_language_labels(lang: str) -> Dict[str, str]:
    """Get language-specific labels"""
    if lang == "中文":
        return {
            "header": "💎 退石最優化計算工具",
            "mode_label": "選擇輸入方式",
            "upload_label": "上傳用石重量 Excel",
            "package_label": "上傳分包資訊 Excel",
            "keyin_label": "直接輸入用石重量",
            "rule_label": "直接輸入分包資訊",
            "stones_label": "用石",
            "result_label": "分配結果",
            "download_label": "下載結果 Excel",
            "error_label": "請上傳正確的 Excel 檔案",
            "info_label": "請上傳檔案或直接輸入資料",
            "no_match": "找不到符合組合",
            "assigned_stones": "分配用石",
            "assigned_weight": "分配重量",
            "expected_weight": "期望重量",
            "diff": "差異值",
            "tolerance": "容許誤差",
            "cts": "cts"
        }
    else:
        return {
            "header": "💎 Stones Returning Optimizer",
            "mode_label": "Select input mode",
            "upload_label": "Upload stones weights Excel",
            "package_label": "Upload packs info Excel",
            "keyin_label": "Key in stones weights",
            "rule_label": "Key in packs info",
            "stones_label": "Stones",
            "result_label": "Result",
            "download_label": "Download result Excel",
            "error_label": "Please upload valid Excel files",
            "info_label": "Please upload files or key in data",
            "no_match": "No match found",
            "assigned_stones": "Assigned stones",
            "assigned_weight": "Assigned Weight",
            "expected_weight": "Expected Weight",
            "diff": "Difference",
            "tolerance": "Tolerance",
            "cts": "cts"
        }

def create_stone_input_grid(labels: Dict[str, str], unique_key: str) -> List[float]:
    """Create stone weight input grid"""
    st.subheader(labels["stones_label"])
    st.markdown(f'<span style="font-size:14px; color:gray;">單位：{labels["cts"]}</span>', 
                unsafe_allow_html=True)
    
    stone_weights = []
    for row in range(6):  # 6 rows x 5 cols = 30 inputs
        cols = st.columns(5)
        for col in range(5):
            idx = row * 5 + col
            with cols[col]:
                st.markdown(f"**{idx+1}.**")
                raw_val = st.text_input(
                    "", 
                    value="", 
                    key=f"stone_{idx}_{unique_key}", 
                    label_visibility="collapsed", 
                    max_chars=10, 
                    placeholder="0.000"
                )
                val = StoneOptimizer.valid_3_decimal(raw_val)
                stone_weights.append(StoneOptimizer.safe_float(val))
    
    return stone_weights

def create_package_rules_input(labels: Dict[str, str], unique_key: str) -> List[Dict]:
    """Create package rules input section"""
    st.subheader(labels["rule_label"])
    
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
                value="", 
                key=f"pcs_{i}_{unique_key}", 
                label_visibility="collapsed", 
                max_chars=3, 
                placeholder="1"
            )
            pcs_val = re.sub(r"\D", "", pcs_raw)[:3] if pcs_raw else "1"
            pcs = int(pcs_val) if pcs_val.isdigit() and int(pcs_val) > 0 else 1
        
        with cols_rule[2]:
            weight_raw = st.text_input(
                "", 
                value="", 
                key=f"weight_{i}_{unique_key}", 
                label_visibility="collapsed", 
                max_chars=10, 
                placeholder="0.000"
            )
            weight_val = StoneOptimizer.valid_3_decimal(weight_raw)
            total_weight = StoneOptimizer.safe_float(weight_val)
        
        with cols_rule[3]:
            pack_id = st.text_input(
                "", 
                value="", 
                key=f"packid_{i}_{unique_key}", 
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
        unique_key = str(uuid.uuid4())
        
        # Stone weights input
        stone_weights = create_stone_input_grid(labels, unique_key)
        
        st.markdown("---")
        
        # Package rules input
        package_rules = create_package_rules_input(labels, unique_key)
        
        st.markdown("---")
        
        # Tolerance input
        tolerance_raw = st.text_input(
            f"{labels['tolerance']}", 
            value="0.003", 
            key=f"tolerance_{unique_key}", 
            placeholder="0.003"
        )
        tolerance_val = StoneOptimizer.valid_3_decimal(tolerance_raw)
        tolerance = StoneOptimizer.safe_float(tolerance_val) or 0.003
        
        st.markdown(f'<div style="text-align:right; color:gray; font-size:14px;">{labels["cts"]}</div>', 
                    unsafe_allow_html=True)
        
        # Calculate results if data is available
        if any(w > 0 for w in stone_weights) and package_rules:
            results = optimizer.calculate_optimal_assignment(
                [w for w in stone_weights if w > 0],  # Filter out zero weights
                package_rules, 
                tolerance, 
                labels
            )
    
    elif mode == labels["upload_label"]:
        # File upload mode
        col1, col2 = st.columns(2)
        
        with col1:
            stone_file = st.file_uploader(labels["upload_label"], type=["xlsx"], key="stone")
        
        with col2:
            package_file = st.file_uploader(labels["package_label"], type=["xlsx"], key="package")
        
        st.markdown("---")
        
        # Tolerance input
        tolerance_raw = st.text_input(
            f"{labels['tolerance']}", 
            value="0.003", 
            key="tolerance_upload", 
            placeholder="0.003"
        )
        tolerance_val = StoneOptimizer.valid_3_decimal(tolerance_raw)
        tolerance = StoneOptimizer.safe_float(tolerance_val) or 0.003
        
        st.markdown(f'<div style="text-align:right; color:gray; font-size:14px;">{labels["cts"]}</div>', 
                    unsafe_allow_html=True)
        
        if stone_file and package_file:
            try:
                # Load Excel files
                stones_df = pd.read_excel(stone_file)
                packages_df = pd.read_excel(package_file)
                
                # Validate columns
                if "cts" not in stones_df.columns:
                    st.error(labels["error_label"])
                    st.stop()
                
                required_cols = ["pcs", "cts"]
                if not all(col in packages_df.columns for col in required_cols):
                    st.error(labels["error_label"])
                    st.stop()
                
                # Extract data
                stones = [w for w in stones_df["cts"].tolist() if w > 0]  # Filter positive weights
                
                package_rules = []
                for _, row in packages_df.iterrows():
                    if pd.notnull(row["pcs"]) and pd.notnull(row["cts"]) and row["pcs"] > 0 and row["cts"] > 0:
                        rule_dict = {
                            "pcs": int(row["pcs"]),
                            "cts": float(row["cts"])
                        }
                        if "Ref" in packages_df.columns and pd.notnull(row["Ref"]) and str(row["Ref"]).strip():
                            rule_dict["Ref"] = str(row["Ref"]).strip()
                        package_rules.append(rule_dict)
                
                # Calculate results
                if stones and package_rules:
                    results = optimizer.calculate_optimal_assignment(stones, package_rules, tolerance, labels)
                
            except Exception as e:
                st.error(f"{labels['error_label']}: {str(e)}")
        else:
            st.info(labels["info_label"])
    
    # Display results
    if results:
        st.markdown("---")
        st.subheader(labels["result_label"])
        
        # Create DataFrame and display
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Download functionality
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        buffer.seek(0)
        
        st.download_button(
            label=labels["download_label"],
            data=buffer,
            file_name="stone_optimization_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Display statistics
        with st.expander("📊 Statistics"):
            matched_count = sum(1 for r in results if r[labels["assigned_stones"]] != labels["no_match"])
            total_count = len(results)
            st.metric("Successful Matches", f"{matched_count}/{total_count}")
            
            if matched_count > 0:
                avg_diff = sum(float(r[labels["diff"]]) for r in results 
                             if r[labels["diff"]] != "-") / matched_count
                st.metric("Average Difference", f"{avg_diff:.3f} cts")

if __name__ == "__main__":
    main()
