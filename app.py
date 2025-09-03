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
    pcs_label = "顆數"
    weight_label = "總重"
    result_label = "分配結果"
    download_label = "下載結果 Excel"
    error_label = "請上傳正確的 Excel 檔案"
    tolerance_label = "容許誤差 (ct)"
    info_label = "請上傳檔案或直接輸入資料"
    no_match = "找不到符合組合"
else:
    st.header("💎 Stones Returning Optimizer")
    mode_label = "Select input mode"
    upload_label = "Upload stones weights Excel"
    package_label = "Upload packs info Excel"
    keyin_label = "Key in stones weights"
    rule_label = "Key in packs info"
    pcs_label = "Pieces"
    weight_label = "Total weight"
    result_label = "Result"
    download_label = "Download result Excel"
    error_label = "Please upload valid Excel files"
    tolerance_label = "Tolerance (ct)"
    info_label = "Please upload files or key in data"
    no_match = "No match found"

st.markdown("---")

# 選擇輸入方式
mode = st.radio(mode_label, [upload_label, keyin_label])

tolerance = st.number_input(tolerance_label, value=0.003, step=0.001, format="%.3f")

results = []

if mode == upload_label:
    # 上傳檔案模式
    diamond_file = st.file_uploader(upload_label, type=["xlsx"], key="diamond")
    package_file = st.file_uploader(package_label, type=["xlsx"], key="package")

    if diamond_file and package_file:
        try:
            diamonds_df = pd.read_excel(diamond_file)
            packages_df = pd.read_excel(package_file)
            diamonds = diamonds_df['重量'].tolist()
            used_indices = set()

            for idx, row in packages_df.iterrows():
                count = int(row['顆數'])
                target = float(row['總重'])
                found = False

                available = [i for i in range(len(diamonds)) if i not in used_indices]
                for combo_indices in itertools.combinations(available, count):
                    combo = [diamonds[i] for i in combo_indices]
                    if abs(sum(combo) - target) <= tolerance:
                        results.append({
                            "分包編號": row['用石編號'],
                            "分配鑽石": combo,
                            "總重": sum(combo)
                        })
                        used_indices.update(combo_indices)
                        found = True
                        break

                if not found:
                    results.append({
                        "分包編號": row['用石編號'],
                        "分配鑽石": no_match,
                        "總重": "-"
                    })
        except Exception as e:
            st.error(error_label)
    else:
        st.info(info_label)

elif mode == keyin_label:
    # 直接輸入模式
    st.subheader(keyin_label)
    diamond_weights = []
    cols = st.columns(5)
    for i in range(30):
        with cols[i % 5]:
            weight = st.number_input(f"鑽石{i+1}", min_value=0.0, step=0.001, format="%.3f", key=f"diamond_{i}")
            diamond_weights.append(weight)

    st.markdown("---")
    st.subheader(rule_label)
    package_rules = []
    for i in range(10):
        col1, col2 = st.columns(2)
        with col1:
            pcs = st.number_input(f"第{i+1}包{pcs_label}", min_value=1, step=1, key=f"pcs_{i}")
        with col2:
            total_weight = st.number_input(f"第{i+1}包{weight_label}", min_value=0.0, step=0.001, format="%.3f", key=f"weight_{i}")
        package_rules.append({"顆數": pcs, "總重": total_weight, "用石編號": i+1})

    if st.button("開始分配" if lang == "中文" else "Start"):
        diamonds = diamond_weights
        used_indices = set()
        for idx, rule in enumerate(package_rules):
            count = int(rule['顆數'])
            target = float(rule['總重'])
            found = False

            available = [i for i in range(len(diamonds)) if i not in used_indices]
            for combo_indices in itertools.combinations(available, count):
                combo = [diamonds[i] for i in combo_indices]
                if abs(sum(combo) - target) <= tolerance:
                    results.append({
                        "分包編號": rule['用石編號'],
                        "分配鑽石": combo,
                        "總重": sum(combo)
                    })
                    used_indices.update(combo_indices)
                    found = True
                    break

            if not found:
                results.append({
                    "分包編號": rule['用石編號'],
                    "分配鑽石": no_match,
                    "總重": "-"
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
