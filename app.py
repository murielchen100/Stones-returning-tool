import streamlit as st
import pandas as pd
import itertools
import io
import re

st.set_page_config(page_title="退石最優化計算工具", layout="wide")
st.image("https://cdn-icons-png.flaticon.com/512/616/616490.png", width=80)

# 語言切換與欄位名稱
lang = st.selectbox("選擇語言 / Language", ["中文", "English"])
if lang == "中文":
    st.header("💎 退石最優化計算工具")
    st.markdown('<div style="font-size:12px; color:gray; margin-bottom:10px;">by Muriel</div>', unsafe_allow_html=True)
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
else:
    st.header("💎 Stones Returning Optimizer")
    st.markdown('<div style="font-size:12px; color:gray; margin-bottom:10px;">by Muriel</div>', unsafe_allow_html=True)
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

def limit_3_decimal(val):
    # 只允許數字、小數點，且最多3位小數
    match = re.match(r"^(\d+)(\.\d{0,3})?$", val)
    if match:
        return match.group(0)
    elif val == "":
        return ""
    else:
        try:
            f = float(val)
            return "{:.3f}".format(f)
        except:
            return ""

def calc_results(stones, package_rules, tolerance, col_pcs, col_weight, col_ref, assigned_stones_label, assigned_weight_label, expected_weight_label, diff_label, no_match):
    results = []
    used_indices = set()
    for idx, rule in enumerate(package_rules):
        count = int(rule[col_pcs])
        target = float(rule[col_weight])
        pack_id = rule[col_ref] if rule[col_ref] else str(idx+1)
        found = False

        available = [i for i in range(len(stones)) if i not in used_indices]
        for combo_indices in itertools.combinations(available, count):
            combo = [stones[i] for i in combo_indices]
            total_assigned = sum(combo)
            diff = abs(total_assigned - target)
            if diff <= tolerance:
                results.append({
                    col_ref: pack_id,
                    assigned_stones_label: combo,
                    assigned_weight_label: "{:.3f}".format(total_assigned),
                    expected_weight_label: "{:.3f}".format(target),
                    diff_label: "{:.3f}".format(diff)
                })
                used_indices.update(combo_indices)
                found = True
                break

        if not found:
            results.append({
                col_ref: pack_id,
                assigned_stones_label: no_match,
                assigned_weight_label: "-",
                expected_weight_label: "{:.3f}".format(target),
                diff_label: "-"
            })
    return results

results = []

if mode == keyin_label:
    st.subheader(stones_label)
    clear_stones = st.button(clear_all_label, key="clear_stones")
    stone_weights = []
    for row in range(6):  # 6 rows x 5 cols = 30
        cols = st.columns(5)
        for col in range(5):
            idx = row * 5 + col
            if idx < 30:
                with cols[col]:
                    st.write(f"{idx+1}.", inline=True)
                    if clear_stones:
                        st.session_state[f"stone_{idx}"] = ""
                    raw_val = st.text_input(
                        "", value=st.session_state.get(f"stone_{idx}", ""), key=f"stone_{idx}", label_visibility="collapsed", max_chars=10
                    )
                    val = limit_3_decimal(raw_val)
                    if val != raw_val:
                        st.session_state[f"stone_{idx}"] = val
                    stone_weights.append(safe_float(val))

    st.markdown("---")
    st.subheader(rule_label)
    clear_rules = st.button(clear_all_label, key="clear_rules")
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
            if clear_rules:
                st.session_state[f"pcs_{i}"] = ""
            pcs_raw = st.text_input("", value=st.session_state.get(f"pcs_{i}", ""), key=f"pcs_{i}", label_visibility="collapsed", max_chars=5)
            pcs_val = limit_3_decimal(pcs_raw)
            if pcs_val != pcs_raw:
                st.session_state[f"pcs_{i}"] = pcs_val
            pcs = int(float(pcs_val)) if pcs_val.replace('.', '', 1).isdigit() and float(pcs_val) > 0 else 1
        with cols_rule[2]:
            if clear_rules:
                st.session_state[f"weight_{i}"] = ""
            weight_raw = st.text_input("", value=st.session_state.get(f"weight_{i}", ""), key=f"weight_{i}", label_visibility="collapsed", max_chars=10)
            weight_val = limit_3_decimal(weight_raw)
            if weight_val != weight_raw:
                st.session_state[f"weight_{i}"] = weight_val
            total_weight = safe_float(weight_val)
        with cols_rule[3]:
            if clear_rules:
                st.session_state[f"packid_{i}"] = ""
            pack_id = st.text_input("", value=st.session_state.get(f"packid_{i}", ""), key=f"packid_{i}", label_visibility="collapsed")
        package_rules.append({
            col_pcs: pcs,
            col_weight: total_weight,
            col_ref: pack_id.strip() if pack_id.strip() else str(i+1)
        })

    st.markdown("---")
    tolerance_raw = st.text_input("容許誤差 (ct) / Tolerance", value="0.003", key="tolerance")
    tolerance_val = limit_3_decimal(tolerance_raw)
    if tolerance_val != tolerance_raw:
        st.session_state["tolerance"] = tolerance_val
    try:
        tolerance = float(tolerance_val)
    except:
        tolerance = 0.003

    if any(stone_weights) and any([r[col_pcs] for r in package_rules]):
        results = calc_results(
            stone_weights, package_rules, tolerance,
            col_pcs, col_weight, col_ref,
            assigned_stones_label, assigned_weight_label, expected_weight_label, diff_label, no_match
        )

elif mode == upload_label:
    stone_file = st.file_uploader(upload_label, type=["xlsx"], key="stone")
    package_file = st.file_uploader(package_label, type=["xlsx"], key="package")
    st.markdown("---")
    tolerance_raw = st.text_input("容許誤差 (ct) / Tolerance", value="0.003", key="tolerance")
    tolerance_val = limit_3_decimal(tolerance_raw)
    if tolerance_val != tolerance_raw:
        st.session_state["tolerance"] = tolerance_val
    try:
        tolerance = float(tolerance_val)
    except:
        tolerance = 0.003

    if stone_file and package_file:
        try:
            stones_df = pd.read_excel(stone_file)
            packages_df = pd.read_excel(package_file)
            stones = stones_df[col_weight].tolist()
            package_rules = []
            for idx, row in packages_df.iterrows():
                package_rules.append({
                    col_pcs: int(row[col_pcs]),
                    col_weight: float(row[col_weight]),
                    col_ref: str(row[col_ref]) if pd.notnull(row[col_ref]) and str(row[col_ref]).strip() else str(idx+1)
                })
            results = calc_results(
                stones, package_rules, tolerance,
                col_pcs, col_weight, col_ref,
                assigned_stones_label, assigned_weight_label, expected_weight_label, diff_label, no_match
            )
        except Exception as e:
            st.error(error_label)
    else:
        st.info(info_label)

# 顯示結果與下載（期望重量靠左顯示）
if results:
    st.subheader(result_label)
    df = pd.DataFrame(results)
    # 讓期望重量欄位轉成字串，避免自動靠右
    if expected_weight_label in df.columns:
        df[expected_weight_label] = df[expected_weight_label].astype(str)
    st.dataframe(df)

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label=download_label,
        data=buffer,
        file_name="result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

