import streamlit as st
import pandas as pd
import io

# 1. 美觀：設定頁面、標題、Logo、分隔線
st.set_page_config(page_title="鑽石分包最佳化工具", page_icon="💎", layout="wide")
st.image("https://cdn-icons-png.flaticon.com/512/616/616490.png", width=80)  # 可換成自己的Logo

# 5. 多語言介面
lang = st.selectbox("選擇語言 / Language", ["中文", "English"])

if lang == "中文":
    st.header("💎 鑽石分包最佳化工具")
    upload_label = "上傳鑽石重量 Excel"
    package_label = "上傳分包規定 Excel"
    result_label = "分配結果"
    download_label = "下載結果 Excel"
    error_label = "請上傳正確的 Excel 檔案"
else:
    st.header("💎 Diamond Packing Optimizer")
    upload_label = "Upload diamond weights Excel"
    package_label = "Upload package rules Excel"
    result_label = "Result"
    download_label = "Download result Excel"
    error_label = "Please upload valid Excel files"

st.markdown("---")

# 3. 上傳檔案
diamond_file = st.file_uploader(upload_label, type=["xlsx"], key="diamond")
package_file = st.file_uploader(package_label, type=["xlsx"], key="package")

tolerance = st.number_input("容許誤差 (ct) / Tolerance", value=0.003, step=0.001, format="%.3f")

results = []

if diamond_file and package_file:
    try:
        diamonds_df = pd.read_excel(diamond_file)
        packages_df = pd.read_excel(package_file)

        # 這裡假設欄位名稱為「重量」「顆數」「總重」「包裝編號」
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
                        "分包編號": row['包裝編號'],
                        "分配鑽石": combo,
                        "總重": sum(combo)
                    })
                    used_indices.update(combo_indices)
                    found = True
                    break

            if not found:
                results.append({
                    "分包編號": row['包裝編號'],
                    "分配鑽石": "找不到符合組合" if lang == "中文" else "No match found",
                    "總重": "-"
                })

        # 1. 美觀：用表格顯示結果
        st.subheader(result_label)
        df = pd.DataFrame(results)
        st.dataframe(df)

        # 2. 下載功能
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label=download_label,
            data=buffer,
            file_name="result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(error_label)
else:
    st.info("請上傳檔案 / Please upload files")
