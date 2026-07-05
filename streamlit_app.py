# -*- coding: utf-8 -*-
"""网盟日报分析 - Streamlit 线上部署入口（Streamlit Cloud 默认识别 streamlit_app.py）"""
from __future__ import annotations

import io
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wangmeng_daily_report import (
    attach_reject_to_sum,
    fmt_date,
    get_analysis_dates,
    load_data,
    normalize_date,
    process_event,
    run as run_report,
)

REQUIRED_SHEETS = ["sum", "event", "reject info"]


def _match_sheet_name(sheet_names: list[str], expected: str) -> str | None:
    lower_map = {s.lower(): s for s in sheet_names}
    return lower_map.get(expected.lower())


def validate_upload_bytes(data: bytes) -> tuple[bool, str, list[date]]:
    try:
        xl = pd.ExcelFile(io.BytesIO(data))
    except Exception as e:
        return False, f"无法读取 Excel：{e}", []

    missing = [s for s in REQUIRED_SHEETS if _match_sheet_name(xl.sheet_names, s) is None]
    if missing:
        return False, f"缺少必要 sheet：{', '.join(missing)}（需要 sum / event / reject info）", []

    try:
        sum_sheet = _match_sheet_name(xl.sheet_names, "sum")
        sum_df = pd.read_excel(xl, sheet_name=sum_sheet)
        if "Time" not in sum_df.columns:
            return False, "sum 表缺少 Time 列", []
        dates = sorted(normalize_date(sum_df["Time"]).dropna().unique())
        if len(dates) < 2:
            return False, "sum 表至少需要 2 个不同日期", []
        return True, "", dates
    except Exception as e:
        return False, f"校验 sum 表失败：{e}", []


def run_analysis_to_bytes(
    data: bytes,
    upload_name: str,
    date_old: date | None,
    date_new: date | None,
) -> tuple[bytes, str, str]:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f_in:
        f_in.write(data)
        input_path = Path(f_in.name)

    output_path = input_path.with_name(input_path.stem + "_out.xlsx")
    try:
        run_report(input_path, output_path, date_old=date_old, date_new=date_new)
        result_bytes = output_path.read_bytes()

        sum_df, event_df, reject_df = load_data(input_path)
        _, reject_counts = process_event(event_df, reject_df)
        sum_processed = attach_reject_to_sum(sum_df, reject_counts)
        d_old, d_new = get_analysis_dates(sum_processed, date_old, date_new)
        period = f"{fmt_date(d_old)} → {fmt_date(d_new)}"
        filename = f"{Path(upload_name).stem}--分析结果.xlsx"
        return result_bytes, period, filename
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


st.set_page_config(page_title="网盟日报分析", layout="wide")
st.title("网盟日报分析")
st.caption("上传包含 sum / event / reject info 三个 sheet 的 Excel，自动生成日报分析结果。")

with st.expander("数据文件要求", expanded=False):
    st.markdown(
        """
- **sum**：流水汇总（含 source、Time、Offer ID、Advertiser、Affiliate 等）
- **event**：事件明细（含 Time、Offer Name、Event、Advertiser、Affiliate）
- **reject info**：reject 规则列表（含 reject info 列）

不勾选「自定义对比日期」时，默认对比数据中**最新两天**。
        """
    )

uploaded = st.file_uploader(
    "上传 Excel 数据文件",
    type=["xlsx", "xls"],
    help="需包含 sum、event、reject info 三个 sheet",
)

if not uploaded:
    st.info("请先上传 Excel 文件。")
    st.stop()

file_bytes = uploaded.getvalue()
ok, err_msg, available_dates = validate_upload_bytes(file_bytes)
if not ok:
    st.error(err_msg)
    st.stop()

date_labels = [fmt_date(d) for d in available_dates]
st.success(f"文件校验通过，共 {len(available_dates)} 个日期：{date_labels[0]} ~ {date_labels[-1]}")

col1, col2, col3 = st.columns(3)
with col1:
    use_custom_dates = st.checkbox("自定义对比日期", value=False)
with col2:
    date_new_label = st.selectbox(
        "新日期（最近一天）",
        date_labels,
        index=len(date_labels) - 1,
        disabled=not use_custom_dates,
    )
with col3:
    date_old_label = st.selectbox(
        "旧日期（对比前一天）",
        date_labels,
        index=len(date_labels) - 2,
        disabled=not use_custom_dates,
    )

date_new = pd.to_datetime(date_new_label).date() if use_custom_dates else None
date_old = pd.to_datetime(date_old_label).date() if use_custom_dates else None

if use_custom_dates and date_old >= date_new:
    st.warning("旧日期必须早于新日期。")
    st.stop()

st.markdown("---")

if "result_bytes" not in st.session_state:
    st.session_state.result_bytes = None
    st.session_state.result_filename = "网盟日报--分析结果.xlsx"
    st.session_state.analysis_period = ""

if st.button("开始分析", type="primary", use_container_width=True):
    with st.spinner("正在计算，数据量较大时可能需要 1~2 分钟…"):
        try:
            result_bytes, period, filename = run_analysis_to_bytes(
                file_bytes, uploaded.name, date_old, date_new
            )
            st.session_state.result_bytes = result_bytes
            st.session_state.analysis_period = period
            st.session_state.result_filename = filename
        except Exception as e:
            st.error(f"分析失败：{e}")
            st.exception(e)

if st.session_state.result_bytes:
    st.success(f"分析完成（对比周期：{st.session_state.analysis_period}）")

    try:
        preview = pd.read_excel(
            io.BytesIO(st.session_state.result_bytes),
            sheet_name="0-整体结论",
        )
        st.subheader("0-整体结论 预览")
        st.dataframe(preview, use_container_width=True, hide_index=True)
    except Exception:
        pass

    st.download_button(
        label="下载分析结果 Excel",
        data=st.session_state.result_bytes,
        file_name=st.session_state.result_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )

    st.caption("结果包含：整体结论、按日汇总、预算波动、分广告主/Affiliate 等 9 个 sheet。")
