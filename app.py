import streamlit as st
import pandas as pd
import itertools
import io
import re
import math

st.set_page_config(page_title="退石最優化計算工具", layout="wide")
st.image("https://cdn-icons-png.flaticon.com/512/616/616490.png", width=80)

lang = st.selectbox("選擇語言 / Language", ["中文", "English"])
if lang == "中文":
    st.header("💎 退石最優化計算工具")
    st.markdown('<div style="font-size:18px; color:green; margin-bottom:10px;">by Muriel</div>', unsafe_allow_html=True)
    mode_label = "選擇輸入方式"
    upload_label = "上傳用石重量 Excel"
    package_label = "上傳分包資訊 Excel"
    keyin_label = "直接輸入用石重量"
    rule_label = "直接輸入分包資訊"
    stones_label = "用石"
    result_label = "分配結果"
    download_label = "下載結果 Excel"
    error_label = "請上傳正確的 Excel 檔案"
    info_label = "請上傳檔案或直接輸入資料"
    no_match = "找不到符合組合"
    assigned_stones_label = "分配用石"
    clear_all_label = "清除全部"
    assigned_weight_label = "分配重量"
    expected_weight_label = "期望重量"
    diff_label = "差異值"
    tolerance_label = "容許誤差"
    cts_label = "cts"
else:
    st.header("💎 Stones Returning Optimizer")
    st.markdown('<div style="font-size:18px; color:green; margin-bottom:10px;">by Muriel</div>', unsafe_allow_html=True)
    mode_label = "Select input mode"
    upload_label = "Upload stones weights Excel"
    package_label = "Upload packs info Excel"
    keyin_label = "Key in stones weights"
    rule_label = "Key in packs info"
    stones_label = "Stones"
    result_label = "Result"
    download_label = "Download result Excel"
    error_label = "Please upload valid Excel files"
    info_label = "Please upload files or key in data"
    no_match = "No match found"
    assigned_stones_label = "Assigned stones"
    clear_all_label = "Clear all"
    assigned_weight_label = "Assigned Weight"
    expected_weight_label = "Expected Weight"
    diff_label = "Difference"
    tolerance_label = "Tolerance"
    cts_label = "cts"

col_pcs = "pcs"
col_weight = "cts"
col_ref = "Ref"

st.markdown("---")

mode = st.radio(mode_label, [upload_label, keyin_label])

def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

def valid_3_decimal(val):
    try:
        f = float(val)
        s = str(f)
        if '.' in s:
            int_part, dec_part = s.split('.')
            return int_part + '.' + dec_part[:3]
        else:
            return s
    except:
        return ""

def truncate_3_decimal(val):
    try:
        f = float(val)
        truncated = math.trunc(f * 1000) / 1000
        return "{:.3f}".format(truncated)
    except:
        return ""

def calc_results(stones, package_rules, tolerance, col_pcs, col_weight, col_ref, assigned_stones_label, assigned_weight_label, expected_weight_label, diff_label, no_match):
    results = []
    used_indices = set()
    for idx, rule in enumerate(package_rules):
        count = int(rule[col_pcs])
        target = float(rule[col_weight])
        pack_id = rule.get(col_ref, "")
        found = False

        available = [i for i in range(len(stones)) if i not in used_indices]
        for combo_indices in itertools.combinations(available, count):
            combo = [stones[i] for i in combo_indices]
            total_assigned = math.trunc(sum(combo) * 1000) / 1000
            diff = math.trunc(abs(total_assigned - target) * 1000) / 1000
            if diff <= tolerance:
                result_row = {
                    assigned_stones_label: combo,
                    assigned_weight_label: "{:.3f}".format(total_assigned),
                    expected_weight_label: "{:.3f}".format(math.trunc(target * 1000) / 1000),
                    diff_label: "{:.3f}".format(diff)
                }
                if pack_id:
                    result_row[col_ref] = pack_id
                results.append(result_row)
                used_indices.update(combo_indices)
                found = True
                break

        if not found:
            result_row = {
                assigned_stones_label: no_match,
                assigned_weight_label: "-",
                expected_weight_label: "{:.3f}".format(math.trunc(target * 1000) / 1000),
                diff_label: "-"
            }
            if pack_id:
                result_row[col_ref] = pack_id
            results.append(result_row)
    return results

results = []

if mode == keyin_label:
    st.subheader(stones_label)
    st.markdown(f'<span style="font-size:14px; color:gray;">單位：{cts_label}</span>', unsafe_allow_html=True)
    # 直接用 value="" 並且 key 每次都不同，確保不會有歷史紀錄
    import uuid
    unique_key = str(uuid.uuid4())
    stone_weights = []
    for row in range(6):  # 6 rows x 5 cols = 30
        cols = st.columns(5)
        for col in range(5):
            idx = row * 5 + col
            with cols[col]:
                st.markdown(f"{idx+1}.", unsafe_allow_html=True)
                raw_val = st.text_input(
                    "", value="", key=f"stone_{idx}_{unique_key}", label_visibility="collapsed", max_chars=10, placeholder="0.000"
                )
                val = valid_3_decimal(raw_val)
                stone_weights.append(safe_float(val))

    st.markdown("---")
    st.subheader(rule_label)
    rule_header = st.columns([0.7, 1.5, 1.5, 2])
    with rule_header[0]:
        st.markdown(" ")
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
            st.markdown(f"{i+1}")
        with cols_rule[1]:
            pcs_raw = st.text_input("", value="", key=f"pcs_{i}_{unique_key}", label_visibility="collapsed", max_chars=5, placeholder="1")
            pcs_val = re.sub(r"\D", "", pcs_raw)[:3] if pcs_raw else "1"
            pcs = int(pcs_val) if pcs_val.isdigit() and int(pcs_val) > 0 else 1
        with cols_rule[2]:
            weight_raw = st.text_input("", value="", key=f"weight_{i}_{unique_key}", label_visibility="collapsed", max_chars=10, placeholder="0.000")
            weight_val = valid_3_decimal(weight_raw)
            total_weight = safe_float(weight_val)
        with cols_rule[3]:
            pack_id = st.text_input("", value="", key=f"packid_{i}_{unique_key}", label_visibility="collapsed", max_chars=20, placeholder="")
        rule_dict = {
            col_pcs: pcs,
            col_weight: total_weight
        }
        if pack_id.strip():
            rule_dict[col_ref] = pack_id.strip()
        package_rules.append(rule_dict)

    st.markdown("---")
    tol_key = "tolerance"
    tolerance_raw = st.text_input(f"{tolerance_label}", value="", key=tol_key+unique_key, placeholder="0.003")
    tolerance_val = valid_3_decimal(tolerance_raw)
    try:
        tolerance = float(tolerance_val)
    except:
        tolerance = 0.003

    st.markdown(f'<div style="text-align:right; color:gray; font-size:14px;">{cts_label}</div>', unsafe_allow_html=True)

    if any(stone_weights) and any(r[col_pcs] for r in package_rules):
        results = calc_results(
            stone_weights, package_rules, tolerance,
            col_pcs, col_weight, col_ref,
            assigned_stones_label, assigned_weight_label, expected_weight_label, diff_label, no_match
        )

elif mode == upload_label:
    stone_file = st.file_uploader(upload_label, type=["xlsx"], key="stone")
    package_file = st.file_uploader(package_label, type=["xlsx"], key="package")
    st.markdown("---")
    tolerance_raw = st.text_input(f"{tolerance_label}", value="0.003", key="tolerance_upload", placeholder="0.003")
    tolerance_val = valid_3_decimal(tolerance_raw)
    try:
        tolerance = float(tolerance_val)
    except:
        tolerance = 0.003
    st.markdown(f'<div style="text-align:right; color:gray; font-size:14px;">{cts_label}</div>', unsafe_allow_html=True)

    if stone_file and package_file:
        try:
            stones_df = pd.read_excel(stone_file)
            packages_df = pd.read_excel(package_file)
            # 欄位檢查
            if col_weight not in stones_df.columns:
                st.error(error_label)
                st.stop()
            required_cols = [col_pcs, col_weight]
            if not all(col in packages_df.columns for col in required_cols):
                st.error(error_label)
                st.stop()
            stones = stones_df[col_weight].tolist()
            package_rules = []
            for idx, row in packages_df.iterrows():
                rule_dict = {
                    col_pcs: int(row[col_pcs]),
                    col_weight: float(row[col_weight])
                }
                if col_ref in packages_df.columns and pd.notnull(row[col_ref]) and str(row[col_ref]).strip():
                    rule_dict[col_ref] = str(row[col_ref]).strip()
                package_rules.append(rule_dict)
            results = calc_results(
                stones, package_rules, tolerance,
                col_pcs, col_weight, col_ref,
                assigned_stones_label, assigned_weight_label, expected_weight_label, diff_label, no_match
            )
        except Exception as e:
            st.error(error_label)
    else:
        st.info(info_label)

# 顯示結果與下載
if results:
    st.subheader(result_label)
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label=download_label,
        data=buffer,
        file_name="result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

