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
    tolerance_label = "容許誤差 (ct)"
    info_label = "請上傳檔案或直接輸入資料"
    no_match = "找不到符合組合"
    start_label = "開始分配"
    assigned_stones_label = "分配用石"
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
    tolerance_label = "Tolerance (ct)"
    info_label = "Please upload files or key in data"
    no_match = "No match found"
    start_label = "Start"
    assigned_stones_label = "Assigned stones"

# 表頭欄位（中英文都一樣）
col_pcs = "pcs"
col_weight = "cts"
col_ref = "Ref"

st.markdown("---")

# 選擇輸入方式
mode = st.radio(mode_label, [upload_label, keyin_label])

tolerance = st.number_input(tolerance_label, value=0.003, step=0.001, format="%.3f")

results = []

if mode == upload_label:
    # 上傳檔案模式
    stone_file = st.file_uploader(upload_label, type=["xlsx"], key="stone")
    package_file = st.file_uploader(package_label, type=["xlsx"], key="package")

    if stone_file and package_file:
        try:
            stones_df = pd.read_excel(stone_file)
            packages_df = pd.read_excel(package_file)
            stones = stones_df[col_weight].tolist()
            used_indices = set()

            # 假設 packages_df 有 col_ref, col_pcs, col_weight 三個欄位
            for idx, row in packages_df.iterrows():
                count = int(row[col_pcs])
                target = float(row[col_weight])
                pack_id = str(row[col_ref]) if pd.notnull(row[col_ref]) and str(row[col_ref]).strip() else str(idx+1)
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
        except Exception as e:
            st.error(error_label)
    else:
        st.info(info_label)

elif mode == keyin_label:
    # 直接輸入模式
    st.subheader(stones_label)
    # Stones 輸入區：只顯示一次 Stones，下面 1~30 編號
    st.markdown("**Stones**")
    st.markdown(" | ".join([str(i+1) for i in range(30)]))
    stone_weights = []
    cols = st.columns(30)
    for i in range(30):
        with cols[i]:
            weight = st.number_input("", min_value=0.0, step=0.001, format="%.3f", key=f"stone_{i}")
            stone_weights.append(weight)

    st.markdown("---")
    st.subheader(rule_label)
    # 分包規則表頭
    st.markdown(f"|   | {col_pcs} | {col_weight} | {col_ref} |")
    st.markdown(f"|---|------|------|------|")
    package_rules = []
    for i in range(10):
        cols_rule = st.columns([1, 2, 2, 3])
        with cols_rule[0]:
            st.markdown(f"**{i+1}**")
        with cols_rule[1]:
            pcs = st.number_input("", min_value=1, step=1, key=f"pcs_{i}")
        with cols_rule[2]:
            total_weight = st.number_input("", min_value=0.0, step=0.001, format="%.3f", key=f"weight_{i}")
        with cols_rule[3]:
            pack_id = st.text_input("", key=f"packid_{i}")
        package_rules.append({
            col_pcs: pcs,
            col_weight: total_weight,
            col_ref: pack_id.strip() if pack_id.strip() else str(i+1)
        })

    if st.button(start_label):
        stones = stone_weights
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
