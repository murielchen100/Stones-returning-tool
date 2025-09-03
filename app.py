import streamlit as st
import pandas as pd
import itertools
import io

st.set_page_config(page_title="退石最優化計算工具", page_icon="💎", layout="wide")
st.image("https://cdn-icons-png.flaticon.com/512/616/616490.png", width=80)

# 多語言
lang = st.selectbox("選擇語言 / Language", ["中文", "English"])
if lang == "中文":
    st.header("💎 退石最優化計算工具")
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
else:
    st.header("💎 Stones Returning Optimizer")
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

col_pcs = "pcs"
col_weight = "cts"
col_ref = "Ref"

st.markdown("---")

mode = st.radio(mode_label, [upload_label, keyin_label])

def calc_results(stones, package_rules, tolerance, col_pcs, col_weight, col_ref, assigned_stones_label, no_match):
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
            if abs(sum(combo) - target) <= tolerance:
                results.append({
                    col_ref: pack_id,
                    assigned_stones_label: combo,
                    col_weight: sum(combo)
                })
                used_indices.update(combo_indices)
                found = True
                break

        if not found:
            results.append({
                col_ref: pack_id,
                assigned_stones_label: no_match,
                col_weight: "-"
            })
    return results

results = []

if mode == keyin_label:
    st.subheader(stones_label)
    # 用石資訊區塊「清除全部」按鈕
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
                        st.session_state[f"stone_{idx}"] = 0.0
                    weight = st.number_input(
                        "", min_value=0.0, step=0.001, format="%.3f",
                        key=f"stone_{idx}", label_visibility="collapsed"
                    )
                    stone_weights.append(weight)

    st.markdown("---")
    st.subheader(rule_label)
    # 分袋資訊區塊「清除全部」按鈕
    clear_rules = st.button(clear_all_label, key="clear_rules")
    # 分包規則表頭
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
                st.session_state[f"pcs_{i}"] = 1
            pcs = st.number_input("", min_value=1, step=1, key=f"pcs_{i}", label_visibility="collapsed")
        with cols_rule[2]:
            if clear_rules:
                st.session_state[f"weight_{i}"] = 0.0
            total_weight = st.number_input("", min_value=0.0, step=0.001, format="%.3f", key=f"weight_{i}", label_visibility="collapsed")
        with cols_rule[3]:
            if clear_rules:
                st.session_state[f"packid_{i}"] = ""
            pack_id = st.text_input("", key=f"packid_{i}", label_visibility="collapsed")
        package_rules.append({
            col_pcs: pcs,
            col_weight: total_weight,
            col_ref: pack_id.strip() if pack_id.strip() else str(i+1)
        })

    # 容許誤差在最下方，調整時自動刷新結果
    st.markdown("---")
    tolerance = st.number_input("容許誤差 (ct) / Tolerance", value=0.003, step=0.001, format="%.3f", key="tolerance")

    # 自動計算結果
    if any(stone_weights) and any([r[col_pcs] for r in package_rules]):
        results = calc_results(stone_weights, package_rules, tolerance, col_pcs, col_weight, col_ref, assigned_stones_label, no_match)

elif mode == upload_label:
    stone_file = st.file_uploader(upload_label, type=["xlsx"], key="stone")
    package_file = st.file_uploader(package_label, type=["xlsx"], key="package")
    st.markdown("---")
    tolerance = st.number_input("容許誤差 (ct) / Tolerance", value=0.003, step=0.001, format="%.3f", key="tolerance")

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
            results = calc_results(stones, package_rules, tolerance, col_pcs, col_weight, col_ref, assigned_stones_label, no_match)
        except Exception as e:
            st.error(error_label)
    else:
        st.info(info_label)

# 顯示結果與下載
if results:
    st.subheader(result_label)
    df = pd.DataFrame(results)
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
