import tempfile
from pathlib import Path

import streamlit as st

from merge_trade_reports import merge_and_export_to_bytes


st.set_page_config(page_title="交易紀錄合併與績效報表", layout="wide")

st.title("交易紀錄合併與績效報表")

st.markdown(
    """
上傳多個 Excel 交易報表（TradingView 匯出格式），系統會自動合併並輸出與原檔相同呈現的報表：

- `績效`
- `交易分析`
- `風險 績效比`
- `交易清單`
- `屬性`
"""
)

uploaded_files = st.file_uploader(
    "選取要合併的 Excel 檔案（可多選）",
    type=["xlsx"],
    accept_multiple_files=True,
)

output_name = st.text_input("輸出檔名", value="merged_report.xlsx")

if uploaded_files:
    st.write(f"已選取檔案數量：{len(uploaded_files)}")

if st.button("產生合併報表", type="primary", disabled=not uploaded_files):
    if not output_name.lower().endswith(".xlsx"):
        output_name = output_name + ".xlsx"

    with st.spinner("處理中..."):
        temp_paths = []
        try:
            for uf in uploaded_files:
                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.getbuffer())
                    temp_paths.append(tmp.name)

            report_bytes = merge_and_export_to_bytes(temp_paths)

        finally:
            for p in temp_paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass

    st.success("完成")
    st.download_button(
        label="下載合併報表",
        data=report_bytes,
        file_name=output_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.divider()

with st.expander("進階說明"):
    st.markdown(
        """
- `已支付佣金` 會維持為 `0`（不計手續費）。
- 指標計算以 `交易清單` 中的 `出場` 交易為基礎。
"""
    )
