import streamlit as st
import itertools

st.title("鑽石分包最佳化工具（直接輸入）")

st.subheader("請輸入30顆鑽石的重量")
diamond_weights = []
cols = st.columns(5)
for i in range(30):
    with cols[i % 5]:
        weight = st.number_input(f"鑽石{i+1}", min_value=0.0, step=0.001, format="%.3f", key=f"diamond_{i}")
        diamond_weights.append(weight)

st.markdown("---")
st.subheader("請輸入10組分包規定（每行：顆數、總重）")
package_rules = []
for i in range(10):
    col1, col2 = st.columns(2)
    with col1:
        pcs = st.number_input(f"第{i+1}包顆數", min_value=1, step=1, key=f"pcs_{i}")
    with col2:
        total_weight = st.number_input(f"第{i+1}包總重", min_value=0.0, step=0.001, format="%.3f", key=f"weight_{i}")
    package_rules.append({"顆數": pcs, "總重": total_weight})

tolerance = st.number_input("容許誤差 (ct)", value=0.003, step=0.001, format="%.3f")

if st.button("開始分配"):
    diamonds = diamond_weights
    results = []
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
                    "分包編號": idx+1,
                    "分配鑽石": combo,
                    "總重": sum(combo)
                })
                used_indices.update(combo_indices)
                found = True
                break

        if not found:
            results.append({
                "分包編號": idx+1,
                "分配鑽石": "找不到符合組合",
                "總重": "-"
            })

    st.write("分配結果：")
    for res in results:
        if isinstance(res['總重'], float):
            st.write(f"分包{res['分包編號']}：{res['分配鑽石']}，總重：{res['總重']:.3f}")
        else:
            st.write(f"分包{res['分包編號']}：{res['分配鑽石']}，總重：{res['總重']}")
