import streamlit as st
import pandas as pd
import itertools
import io

st.set_page_config(page_title="退石最優化計算工具", page_icon="💎", layout="wide")
st.image("https://cdn-icons-png.flaticon.com/512/616/616490.png", width=80)

# ...（語言切換區略）

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
                    assigned_weight_label: f"{total_assigned:.4f}",
                    expected_weight_label: f"{target:.4f}",
                    diff_label: f"{diff:.4f}"
                })
                used_indices.update(combo_indices)
                found = True
                break

        if not found:
            results.append({
                col_ref: pack_id,
                assigned_stones_label: no_match,
                assigned_weight_label: "-",
                expected_weight_label: f"{target:.4f}",
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
                    val = st.text_input(
                        "", value=st.session_state.get(f"stone_{idx}", ""), key=f"stone_{idx}", label_visibility="collapsed", max_chars=10
                    )
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
            pcs_val = st.text_input("", value=st.session_state.get(f"pcs_{i}", ""), key=f"pcs_{i}", label_visibility="collapsed", max_chars=5)
            pcs = int(pcs_val) if pcs_val.isdigit() and int(pcs_val) > 0 else 1
        with cols_rule[2]:
            if clear_rules:
                st.session_state[f"weight_{i}"] = ""
            weight_val = st.text_input("", value=st.session_state.get(f"weight_{i}", ""), key=f"weight_{i}", label_visibility="collapsed", max_chars=10)
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
    tolerance_val = st.text_input("容許誤差 (ct) / Tolerance", value="0.003", key="tolerance")
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
    tolerance_val = st.text_input("容許誤差 (ct) / Tolerance", value="0.003", key="tolerance")
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
