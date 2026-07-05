# -*- coding: utf-8 -*-
"""
网盟日报分析
输入 Excel（sum / event / reject info），按需求文档计算并输出分析结果 Excel。
"""
from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BUDGET_KEYS = ["source", "Advertiser", "Offer ID", "Adv Offer ID", "App ID", "GEO"]
APPNEXT_ADVERTISERS = {"[20]appnext"}
CONCLUSION_BUDGET_PROFIT_THRESHOLD = 10  # 0-整体结论：仅展示 |利润变化| >= 10 美金的预算


def safe_div(a, b, default=0.0):
    a, b = float(a), float(b)
    return a / b if b != 0 else default


def pct_change(new, old):
    new, old = float(new), float(old)
    if old == 0:
        return np.nan
    return (new - old) / old


def day_has_activity(df_day: pd.DataFrame) -> bool:
    """该自然日 sum 中是否存在可对比的业务数据。"""
    if df_day.empty:
        return False
    return (
        float(df_day["Total Revenue"].sum()) > 0
        or float(df_day["Total Profit"].sum()) != 0
        or float(df_day["Total Clicks"].sum()) > 0
    )


def merge_with_day_flags(old_df: pd.DataFrame, new_df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """合并两日聚合表，并标记各维度在前/后一日是否出现过。"""
    merged = old_df.merge(new_df, on=keys, how="outer", suffixes=("_old", "_new"))
    merged["had_old"] = merged["Total_Revenue_old"].notna() | merged["Total_Profit_old"].notna()
    merged["had_new"] = merged["Total_Revenue_new"].notna() | merged["Total_Profit_new"].notna()
    for c in merged.columns:
        if c.endswith("_old") or c.endswith("_new"):
            if c in keys or c in ("had_old", "had_new"):
                continue
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)
    return merged


def fmt_old_day_value(val, had_old: bool, *, as_rate: bool = False):
    """前日无记录时展示 —，有记录时展示数值或百分比。"""
    if not had_old:
        return "—"
    if as_rate:
        if pd.isna(val):
            return "N/A"
        return fmt_pct(val, digits=2)
    return round(float(val), 2)


def fmt_pct_change_display(
    new,
    old,
    had_old: bool,
    had_new: bool = True,
    *,
    allow_negative: bool = True,
) -> str:
    """
    日环比展示：前日无数据→新增；前日有、当日无→停投；双日有数→百分比。
    """
    new_v = 0.0 if pd.isna(new) else float(new)
    old_v = 0.0 if pd.isna(old) else float(old)

    if not had_old and had_new:
        if new_v > 0 or (allow_negative and new_v < 0):
            return "新增"
        return "—"
    if had_old and not had_new:
        if old_v > 0 or (allow_negative and old_v < 0):
            return "停投"
        return "—"
    if not had_old and not had_new:
        return "—"
    if old_v == 0:
        if new_v > 0 or (allow_negative and new_v < 0):
            return "新增"
        return "—"
    if new_v == 0 and old_v != 0:
        return "停投"
    pct = pct_change(new_v, old_v)
    if pd.isna(pct):
        return "—"
    return f"{pct * 100:.1f}%"


def profit_fluctuation_ratio(
    profit_new: float,
    profit_old: float,
    had_old_day: bool,
) -> float:
    """整体利润波动比例；前日无大盘数据时不应判为 0% 稳定。"""
    if not had_old_day:
        return 1.0 if abs(profit_new) > 0.01 else 0.0
    pct = pct_change(profit_new, profit_old)
    if pd.isna(pct):
        return 1.0 if profit_new != profit_old else 0.0
    return abs(pct)


def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.{digits}f}%"


def fmt_money(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{float(x):.{digits}f}"


def fmt_date(d) -> str:
    if isinstance(d, pd.Timestamp):
        d = d.date()
    if isinstance(d, date):
        return f"{d.year}/{d.month}/{d.day}"
    return str(d)


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def extract_offer_id_from_name(name) -> str:
    if pd.isna(name):
        return ""
    m = re.search(r"\[(\d+)\]", str(name))
    return m.group(1) if m else ""


def is_reject_event(event, reject_set: set[str]) -> bool:
    if pd.isna(event):
        return False
    el = str(event).strip().lower()
    return el in reject_set


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    sum_df = pd.read_excel(path, sheet_name="sum")
    event_df = pd.read_excel(path, sheet_name="event")
    reject_df = pd.read_excel(path, sheet_name="reject info")
    return sum_df, event_df, reject_df


def process_event(event_df: pd.DataFrame, reject_df: pd.DataFrame) -> pd.DataFrame:
    reject_set = set(reject_df["reject info"].astype(str).str.strip().str.lower())
    df = event_df.copy()
    df["是否reject事件"] = df["Event"].map(lambda x: is_reject_event(x, reject_set)).map(
        {True: "reject事件", False: "非reject事件"}
    )
    df["Offer ID"] = df["Offer Name"].map(extract_offer_id_from_name)
    df["Time_date"] = normalize_date(df["Time"])

    # Appnext [20]appnext：reject 事件日期减 1 天
    mask = (df["Advertiser"].astype(str).isin(APPNEXT_ADVERTISERS)) & (
        df["是否reject事件"] == "reject事件"
    )
    df.loc[mask, "Time_date"] = df.loc[mask, "Time_date"].apply(
        lambda d: d - timedelta(days=1) if pd.notna(d) else d
    )

    reject_counts = (
        df[df["是否reject事件"] == "reject事件"]
        .groupby(["Time_date", "Offer ID", "Advertiser", "Affiliate"], dropna=False)
        .size()
        .reset_index(name="reject_count")
    )
    return df, reject_counts


def attach_reject_to_sum(sum_df: pd.DataFrame, reject_counts: pd.DataFrame) -> pd.DataFrame:
    df = sum_df.copy()
    df["Time_date"] = normalize_date(df["Time"])
    df["Offer ID_str"] = df["Offer ID"].astype(str)
    rc = reject_counts.copy()
    rc["Offer ID_str"] = rc["Offer ID"].astype(str)
    rc = rc[["Time_date", "Offer ID_str", "Advertiser", "Affiliate", "reject_count"]]

    merged = df.merge(
        rc,
        on=["Time_date", "Offer ID_str", "Advertiser", "Affiliate"],
        how="left",
    )
    merged["reject_count"] = merged["reject_count"].fillna(0).astype(int)
    return merged


def get_analysis_dates(
    sum_df: pd.DataFrame,
    date_old: date | None = None,
    date_new: date | None = None,
) -> tuple[date, date]:
    available = sorted(sum_df["Time_date"].dropna().unique())
    if len(available) < 2:
        raise ValueError("sum 表至少需要 2 个日期")

    if date_new is not None and date_old is not None:
        if date_old not in available or date_new not in available:
            raise ValueError(
                f"指定日期不在数据中。可用日期：{', '.join(fmt_date(d) for d in available)}"
            )
        if date_old >= date_new:
            raise ValueError("date_old 必须早于 date_new")
        return date_old, date_new

    return available[-2], available[-1]


def daily_summary(sum_df: pd.DataFrame, date_old: date, date_new: date) -> pd.DataFrame:
    daily = (
        sum_df.groupby("Time_date", dropna=False)
        .agg(
            Total_Revenue=("Total Revenue", "sum"),
            Total_Cost=("Total Cost", "sum"),
            Total_Profit=("Total Profit", "sum"),
        )
        .reset_index()
    )
    daily["Total_Profit_Rate"] = daily.apply(
        lambda r: safe_div(r["Total_Profit"], r["Total_Revenue"], np.nan), axis=1
    )
    return daily


def decompose_profit_change(rev_old, rev_new, profit_old, profit_new):
    margin_old = safe_div(profit_old, rev_old, 0.0)
    margin_new = safe_div(profit_new, rev_new, 0.0)
    rev_contrib = (rev_new - rev_old) * margin_old
    margin_contrib = rev_new * (margin_new - margin_old)
    return rev_contrib, margin_contrib, margin_old, margin_new


def decompose_affiliate_factors(row_old, row_new):
    """Profit = Clicks * CR * unit_price * margin"""
    c_old, c_new = float(row_old["clicks"]), float(row_new["clicks"])
    conv_old, conv_new = float(row_old["conversions"]), float(row_new["conversions"])
    r_old, r_new = float(row_old["revenue"]), float(row_new["revenue"])
    p_old, p_new = float(row_old["profit"]), float(row_new["profit"])

    cr_old = safe_div(conv_old, c_old)
    cr_new = safe_div(conv_new, c_new)
    up_old = safe_div(r_old, conv_old)
    up_new = safe_div(r_new, conv_new)
    m_old = safe_div(p_old, r_old)
    m_new = safe_div(p_new, r_new)

    clicks_eff = (c_new - c_old) * cr_old * up_old * m_old
    cr_eff = c_new * (cr_new - cr_old) * up_old * m_old
    up_eff = c_new * cr_new * (up_new - up_old) * m_old
    m_eff = c_new * cr_new * up_new * (m_new - m_old)
    return {
        "clicks": clicks_eff,
        "cr": cr_eff,
        "unit_price": up_eff,
        "margin": m_eff,
        "cr_old": cr_old,
        "cr_new": cr_new,
        "clicks_old": c_old,
        "clicks_new": c_new,
        "up_old": up_old,
        "up_new": up_new,
        "m_old": m_old,
        "m_new": m_new,
    }


def top_factors_text(factors: dict, profit_change: float, threshold=0.8) -> str:
    total = sum(abs(v) for v in factors.values())
    if total < 1e-9:
        return ""
    ranked = sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True)
    picked = []
    cum = 0.0
    for name, val in ranked:
        if abs(val) < 1e-9:
            continue
        picked.append((name, val))
        cum += abs(val) / total
        if cum >= threshold:
            break
    name_map = {
        "clicks": "Total Clicks",
        "cr": "CR",
        "unit_price": "单价",
        "margin": "Total Profit Rate",
    }
    parts = []
    for name, val in picked:
        parts.append(f"{name_map.get(name, name)}{'增加' if val >= 0 else '减少'}利润{abs(val):.2f}美金")
    return "，主因" + "，".join(parts)


def contrib_ratios(rev_contrib: float, margin_contrib: float) -> tuple[float, float]:
    total = abs(rev_contrib) + abs(margin_contrib)
    if total < 1e-9:
        return 0.0, 0.0
    return abs(rev_contrib) / total * 100, abs(margin_contrib) / total * 100


def get_affiliate_profit_impacts(
    sum_df: pd.DataFrame,
    keys: list[str],
    budget_row,
    date_old: date,
    date_new: date,
    profit_change_budget: float,
) -> list[tuple[str, float]]:
    """返回同向变化的 Affiliate 及其利润影响幅度，按影响排序。"""
    mask = pd.Series(True, index=sum_df.index)
    for k in keys:
        mask &= sum_df[k].astype(str) == str(budget_row[k])

    sub = sum_df[mask & sum_df["Time_date"].isin([date_old, date_new])]
    if sub.empty:
        return []

    aff_daily = (
        sub.groupby(["Affiliate", "Time_date"], dropna=False)["Total Profit"]
        .sum()
        .reset_index()
    )

    same_direction = profit_change_budget > 0
    impacts: list[tuple[str, float]] = []

    for aff, grp in aff_daily.groupby("Affiliate"):
        p_old = grp.loc[grp["Time_date"] == date_old, "Total Profit"].sum()
        p_new = grp.loc[grp["Time_date"] == date_new, "Total Profit"].sum()
        p_change = float(p_new - p_old)
        if same_direction and p_change <= 0:
            continue
        if not same_direction and p_change >= 0:
            continue
        if abs(p_change) < 0.01:
            continue
        impacts.append((str(aff), p_change))

    impacts.sort(key=lambda x: x[1], reverse=bool(profit_change_budget > 0))
    return impacts


def build_budget_conclusion_rows(
    sum_df: pd.DataFrame,
    date_old: date,
    date_new: date,
    profit_trend_up: bool,
) -> list[dict]:
    """需求 3-b/c：每个同向预算一行，含 Revenue/Profit Rate 贡献与 Affiliate 影响。"""
    keys = BUDGET_KEYS
    old_df = aggregate_budget(sum_df, keys, date_old)
    new_df = aggregate_budget(sum_df, keys, date_new)
    merged = merge_with_day_flags(old_df, new_df, keys)
    merged["profit_change"] = merged["Total_Profit_new"] - merged["Total_Profit_old"]

    if profit_trend_up:
        candidates = merged[merged["profit_change"] >= CONCLUSION_BUDGET_PROFIT_THRESHOLD].copy()
        candidates = candidates.sort_values("profit_change", ascending=False)
    else:
        candidates = merged[merged["profit_change"] <= -CONCLUSION_BUDGET_PROFIT_THRESHOLD].copy()
        candidates = candidates.sort_values("profit_change", ascending=True)

    rows: list[dict] = []
    for _, r in candidates.iterrows():
        rev_c, margin_c, _, _ = decompose_profit_change(
            r["Total_Revenue_old"],
            r["Total_Revenue_new"],
            r["Total_Profit_old"],
            r["Total_Profit_new"],
        )
        p_change = float(r["profit_change"])

        aff_impacts = get_affiliate_profit_impacts(
            sum_df, keys, r, date_old, date_new, p_change
        )
        if aff_impacts:
            aff_text = "，".join(
                f"{aff}影响利润{pc:.2f}美金" for aff, pc in aff_impacts
            )
        else:
            aff_text = "无"

        budget_key = (
            f"{r['source']}--{r['Offer ID']}--{r['Advertiser']}--{r['App ID']}--{r['GEO']}"
        )
        conclusion = (
            f"{budget_key}，影响利润{p_change:.2f}美金"
            f"(流水影响{rev_c:.2f}美金，利润率影响{margin_c:.2f}美金)，"
            f"该offer下核心Affilate影响：{aff_text}"
        )

        rows.append({"类型": "预算分析", "结论": conclusion})
    return rows


def build_overall_conclusion_sheet(
    sum_df: pd.DataFrame, date_old: date, date_new: date
) -> tuple[pd.DataFrame, dict]:
    """生成 0-整体结论 sheet：首行整体摘要，每个预算单独一行。"""
    d_old = sum_df[sum_df["Time_date"] == date_old]
    d_new = sum_df[sum_df["Time_date"] == date_new]

    had_old_day = day_has_activity(d_old)
    had_new_day = day_has_activity(d_new)

    rev_old = d_old["Total Revenue"].sum()
    rev_new = d_new["Total Revenue"].sum()
    profit_old = d_old["Total Profit"].sum()
    profit_new = d_new["Total Profit"].sum()

    rev_contrib, margin_contrib, margin_old, margin_new = decompose_profit_change(
        rev_old, rev_new, profit_old, profit_new
    )
    profit_change = profit_new - profit_old
    profit_fluct_pct = profit_fluctuation_ratio(profit_new, profit_old, had_old_day)

    def trend_word(delta):
        if delta > 0:
            return "增加"
        if delta < 0:
            return "减少"
        return "持平"

    if not had_old_day and had_new_day:
        base = (
            f"{fmt_date(date_new)}总流水为{rev_new:.2f}美金，{fmt_date(date_old)}无流水数据，"
            f"利润率为{fmt_pct(margin_new, digits=2)}，"
            f"利润为{profit_new:.2f}美金，较前一对比日无可比基准，变化幅度{profit_change:.2f}美金"
        )
    elif had_old_day and not had_new_day:
        base = (
            f"{fmt_date(date_new)}无流水数据，相比于{fmt_date(date_old)}总流水{rev_old:.2f}美金、"
            f"利润{profit_old:.2f}美金，整体停投，变化幅度{profit_change:.2f}美金"
        )
    else:
        rev_pct_str = fmt_pct_change_display(rev_new, rev_old, had_old_day, had_new_day)
        margin_pct_str = fmt_pct_change_display(
            margin_new, margin_old, had_old_day, had_new_day, allow_negative=True
        )
        profit_pct_str = fmt_pct_change_display(
            profit_new, profit_old, had_old_day, had_new_day, allow_negative=True
        )
        base = (
            f"{fmt_date(date_new)}总流水为{rev_new:.2f}美金，相比于{fmt_date(date_old)}"
            f"{trend_word(rev_new - rev_old)}{rev_pct_str}/变化幅度{rev_new - rev_old:.2f}美金，"
            f"利润率为{fmt_pct(margin_new, digits=2)}，相比于{fmt_date(date_old)}"
            f"{trend_word(margin_new - margin_old)}{margin_pct_str}，"
            f"利润为{profit_new:.2f}美金，相比于{fmt_date(date_old)}"
            f"{trend_word(profit_change)}{profit_pct_str}/变化幅度{profit_change:.2f}美金"
        )

    meta = {
        "date_old": date_old,
        "date_new": date_new,
        "profit_change": profit_change,
        "profit_fluct_pct": profit_fluct_pct,
        "rev_contrib": rev_contrib,
        "margin_contrib": margin_contrib,
        "profit_trend_up": profit_change > 0,
    }

    if profit_fluct_pct < 0.05:
        summary = base + "，整体稳定。"
        sheet_rows = [{"类型": "整体结论", "结论": summary}]
        return pd.DataFrame(sheet_rows), meta

    detail = (
        f"，其中流水{trend_word(rev_contrib)}影响利润{trend_word(rev_contrib)}{abs(rev_contrib):.2f}美金，"
        f"利润率{trend_word(margin_contrib)}影响利润{trend_word(margin_contrib)}{abs(margin_contrib):.2f}美金。"
    )
    summary = base + detail
    sheet_rows = [{"类型": "整体结论", "结论": summary}]
    sheet_rows.extend(
        build_budget_conclusion_rows(sum_df, date_old, date_new, profit_change > 0)
    )
    return pd.DataFrame(sheet_rows), meta


def aggregate_budget(sum_df: pd.DataFrame, keys: Iterable[str], day: date) -> pd.DataFrame:
    sub = sum_df[sum_df["Time_date"] == day]
    agg = sub.groupby(list(keys), dropna=False).agg(
        Total_Clicks=("Total Clicks", "sum"),
        Total_Conversions=("Total Conversions", "sum"),
        Total_Revenue=("Total Revenue", "sum"),
        Total_Cost=("Total Cost", "sum"),
        Total_Profit=("Total Profit", "sum"),
        reject=("reject_count", "sum"),
        Status=("Status", "first"),
    ).reset_index()
    agg["CR"] = agg.apply(lambda r: safe_div(r["Total_Conversions"], r["Total_Clicks"]), axis=1)
    agg["Total_Profit_Rate"] = agg.apply(
        lambda r: safe_div(r["Total_Profit"], r["Total_Revenue"], np.nan), axis=1
    )
    return agg


def is_new_budget(sum_df: pd.DataFrame, keys: list[str], budget_row, date_new: date) -> str:
    mask = pd.Series(True, index=sum_df.index)
    for k in keys:
        mask &= sum_df[k].astype(str) == str(budget_row[k])
    hist = sum_df[mask & (sum_df["Total Revenue"] > 0)]
    if hist.empty:
        return "新预算"
    first_date = hist["Time_date"].min()
    return "新预算" if first_date >= date_new - timedelta(days=7) else "旧预算"


def analyze_affiliate_text(
    sum_df: pd.DataFrame,
    keys: list[str],
    budget_row,
    date_old: date,
    date_new: date,
    profit_change_budget: float,
) -> str:
    mask = pd.Series(True, index=sum_df.index)
    for k in keys:
        mask &= sum_df[k].astype(str) == str(budget_row[k])

    sub = sum_df[mask & sum_df["Time_date"].isin([date_old, date_new])]
    if sub.empty:
        return "无Affiliate数据"

    aff_daily = (
        sub.groupby(["Affiliate", "Time_date"], dropna=False)
        .agg(
            clicks=("Total Clicks", "sum"),
            conversions=("Total Conversions", "sum"),
            revenue=("Total Revenue", "sum"),
            profit=("Total Profit", "sum"),
        )
        .reset_index()
    )

    aff_items: list[tuple[float, str]] = []
    same_direction = profit_change_budget > 0

    for aff, grp in aff_daily.groupby("Affiliate"):
        old = grp[grp["Time_date"] == date_old]
        new = grp[grp["Time_date"] == date_new]
        row_old = {
            "clicks": old["clicks"].sum() if not old.empty else 0,
            "conversions": old["conversions"].sum() if not old.empty else 0,
            "revenue": old["revenue"].sum() if not old.empty else 0,
            "profit": old["profit"].sum() if not old.empty else 0,
        }
        row_new = {
            "clicks": new["clicks"].sum() if not new.empty else 0,
            "conversions": new["conversions"].sum() if not new.empty else 0,
            "revenue": new["revenue"].sum() if not new.empty else 0,
            "profit": new["profit"].sum() if not new.empty else 0,
        }
        p_change = row_new["profit"] - row_old["profit"]
        if same_direction and p_change <= 0:
            continue
        if not same_direction and p_change >= 0:
            continue
        if abs(p_change) < 0.01:
            continue

        rev_c, margin_c, _, _ = decompose_profit_change(
            row_old["revenue"], row_new["revenue"], row_old["profit"], row_new["profit"]
        )
        factors = decompose_affiliate_factors(row_old, row_new)
        factor_text = top_factors_text(
            {k: factors[k] for k in ("clicks", "cr", "unit_price", "margin")},
            p_change,
        )
        trend = "增加" if p_change > 0 else "减少"
        extra = ""
        if abs(factors["clicks"]) >= abs(factors["cr"]):
            extra = (
                f"，主因Total Clicks{trend}利润{abs(factors['clicks']):.2f}美金，"
                f"从{row_old['clicks']:.0f}变化到{row_new['clicks']:.0f}"
            )
        elif abs(factors["cr"]) > 0:
            extra = (
                f"，主因CR{trend}利润{abs(factors['cr']):.2f}美金，"
                f"从{fmt_pct(factors['cr_old'])}变化到{fmt_pct(factors['cr_new'])}"
            )

        line = (
            f"{aff}共{trend}利润{abs(p_change):.2f}美金，其中Total Revenue和Total Profit Rate "
            f"分别{'增加' if rev_c >= 0 else '减少'}{abs(rev_c):.2f}美金、"
            f"{'增加' if margin_c >= 0 else '减少'}{abs(margin_c):.2f}美金"
            f"{factor_text or extra}"
        )
        aff_items.append((p_change, line))

    if not aff_items:
        return "无同向变化Affiliate"

    # 利润增加：按影响利润降序；利润减少：按影响利润升序（降幅最大的在前）
    aff_items.sort(key=lambda x: x[0], reverse=bool(profit_change_budget > 0))
    return "\n".join(line for _, line in aff_items)


def build_budget_fluctuation_table(
    sum_df: pd.DataFrame, date_old: date, date_new: date
) -> pd.DataFrame:
    keys = BUDGET_KEYS
    old_df = aggregate_budget(sum_df, keys, date_old)
    new_df = aggregate_budget(sum_df, keys, date_new)

    merged = merge_with_day_flags(old_df, new_df, keys)
    merged["profit_change"] = merged["Total_Profit_new"] - merged["Total_Profit_old"]

    # 利润增加超过5美金 或 利润下降超过5美金 均纳入，按最近一天 Total Revenue 降序
    candidates = merged[merged["profit_change"].abs() >= 5].copy()
    candidates = candidates.sort_values("Total_Revenue_new", ascending=False)

    rows = []
    d_old_s, d_new_s = fmt_date(date_old), fmt_date(date_new)
    for _, r in candidates.iterrows():
        aff_text = analyze_affiliate_text(sum_df, keys, r, date_old, date_new, r["profit_change"])
        rows.append(
            {
                "source": r["source"],
                "Advertiser": r["Advertiser"],
                "Offer ID": r["Offer ID"],
                "Adv Offer ID": r["Adv Offer ID"],
                "App ID": r["App ID"],
                "GEO": r["GEO"],
                "Status": r.get("Status_new") or r.get("Status_old"),
                f"{d_new_s} Total Revenue": r["Total_Revenue_new"],
                f"{d_old_s} Total Revenue": fmt_old_day_value(r["Total_Revenue_old"], r["had_old"]),
                f"{d_new_s} Total Profit Rate": fmt_pct(
                    safe_div(r["Total_Profit_new"], r["Total_Revenue_new"], np.nan), digits=2
                ),
                f"{d_old_s} Total Profit Rate": fmt_old_day_value(
                    safe_div(r["Total_Profit_old"], r["Total_Revenue_old"], np.nan),
                    r["had_old"],
                    as_rate=True,
                ),
                f"{d_new_s} 利润": r["Total_Profit_new"],
                f"{d_old_s} 利润": fmt_old_day_value(r["Total_Profit_old"], r["had_old"]),
                "利润变化幅度": r["profit_change"],
                "分Affiliate影响因素分析": aff_text,
                "是否为新老预算": is_new_budget(sum_df, keys, r, date_new),
            }
        )
    return pd.DataFrame(rows)


def build_budget_detail_table(
    sum_df: pd.DataFrame, date_old: date, date_new: date
) -> pd.DataFrame:
    """需求文档第4节(1)：含 Clicks/CR/Cost 等流量指标的预算波动明细。"""
    keys = BUDGET_KEYS
    old_df = aggregate_budget(sum_df, keys, date_old)
    new_df = aggregate_budget(sum_df, keys, date_new)
    merged = merge_with_day_flags(old_df, new_df, keys)
    merged["profit_change"] = merged["Total_Profit_new"] - merged["Total_Profit_old"]

    candidates = merged[merged["profit_change"].abs() >= 5].copy()
    candidates = candidates.sort_values("Total_Revenue_new", ascending=False)

    rows = []
    for _, r in candidates.iterrows():
        cr_new = safe_div(r["Total_Conversions_new"], r["Total_Clicks_new"])
        rows.append(
            {
                "source": r["source"],
                "Advertiser": r["Advertiser"],
                "Offer ID": r["Offer ID"],
                "Adv Offer ID": r["Adv Offer ID"],
                "App ID": r["App ID"],
                "GEO": r["GEO"],
                "Status": r.get("Status_new") or r.get("Status_old"),
                "Total Clicks": r["Total_Clicks_new"],
                "Total Conversions": r["Total_Conversions_new"],
                "CR": cr_new,
                "Total Cost": r["Total_Cost_new"],
                "Total Profit": r["Total_Profit_new"],
                "利润变化幅度": r["profit_change"],
                "是否为新老预算": is_new_budget(sum_df, keys, r, date_new),
                "分Affiliate影响因素分析": analyze_affiliate_text(
                    sum_df, keys, r, date_old, date_new, r["profit_change"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_advertiser_table(
    sum_df: pd.DataFrame, date_old: date, date_new: date
) -> pd.DataFrame:
    keys = ["source", "Advertiser"]

    def _agg(day):
        sub = sum_df[sum_df["Time_date"] == day]
        g = sub.groupby(keys, dropna=False).agg(
            Total_Revenue=("Total Revenue", "sum"),
            Total_Profit=("Total Profit", "sum"),
            Total_Conversions=("Total Conversions", "sum"),
            reject=("reject_count", "sum"),
        ).reset_index()
        g["Total_Profit_Rate"] = g.apply(
            lambda r: safe_div(r["Total_Profit"], r["Total_Revenue"], np.nan), axis=1
        )
        g["reject_ratio"] = g.apply(
            lambda r: safe_div(r["reject"], r["reject"] + r["Total_Conversions"], np.nan),
            axis=1,
        )
        return g

    old_g, new_g = _agg(date_old), _agg(date_new)
    merged = merge_with_day_flags(old_g, new_g, keys)

    rows = []
    d_old_s, d_new_s = fmt_date(date_old), fmt_date(date_new)
    for _, r in merged.iterrows():
        had_old, had_new = bool(r["had_old"]), bool(r["had_new"])
        rows.append(
            {
                "source": r["source"],
                "Advertiser": r["Advertiser"],
                f"{d_new_s} Total Revenue": r["Total_Revenue_new"],
                f"{d_old_s} Total Revenue": fmt_old_day_value(r["Total_Revenue_old"], had_old),
                "Total Revenue 日环比": fmt_pct_change_display(
                    r["Total_Revenue_new"], r["Total_Revenue_old"], had_old, had_new
                ),
                f"{d_new_s} Total Profit": r["Total_Profit_new"],
                f"{d_old_s} Total Profit": fmt_old_day_value(r["Total_Profit_old"], had_old),
                "Total Profit 日环比": fmt_pct_change_display(
                    r["Total_Profit_new"], r["Total_Profit_old"], had_old, had_new, allow_negative=True
                ),
                f"{d_new_s} Total Profit Rate": fmt_pct(r["Total_Profit_Rate_new"], digits=2),
                f"{d_old_s} Total Profit Rate": fmt_old_day_value(
                    r["Total_Profit_Rate_old"], had_old, as_rate=True
                ),
                "Total Profit Rate 日环比": fmt_pct_change_display(
                    r["Total_Profit_Rate_new"],
                    r["Total_Profit_Rate_old"],
                    had_old,
                    had_new,
                    allow_negative=True,
                ),
                f"{d_new_s} reject比例": fmt_pct(r["reject_ratio_new"], digits=2),
                "reject比例 日环比": fmt_pct_change_display(
                    r["reject_ratio_new"], r["reject_ratio_old"], had_old, had_new
                ),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(f"{d_new_s} Total Revenue", ascending=False)
    return df


def build_affiliate_table(
    sum_df: pd.DataFrame, date_old: date, date_new: date
) -> pd.DataFrame:
    keys = ["source", "Affiliate"]

    def _agg(day):
        sub = sum_df[sum_df["Time_date"] == day]
        g = sub.groupby(keys, dropna=False).agg(
            Total_Revenue=("Total Revenue", "sum"),
            Total_Profit=("Total Profit", "sum"),
            Total_Conversions=("Total Conversions", "sum"),
            reject=("reject_count", "sum"),
        ).reset_index()
        g["Total_Profit_Rate"] = g.apply(
            lambda r: safe_div(r["Total_Profit"], r["Total_Revenue"], np.nan), axis=1
        )
        g["reject_ratio"] = g.apply(
            lambda r: safe_div(r["reject"], r["reject"] + r["Total_Conversions"], np.nan),
            axis=1,
        )
        return g

    old_g, new_g = _agg(date_old), _agg(date_new)
    merged = merge_with_day_flags(old_g, new_g, keys)

    rows = []
    d_old_s, d_new_s = fmt_date(date_old), fmt_date(date_new)
    for _, r in merged.iterrows():
        had_old, had_new = bool(r["had_old"]), bool(r["had_new"])
        rows.append(
            {
                "source": r["source"],
                "Affiliate": r["Affiliate"],
                f"{d_new_s} Total Revenue": r["Total_Revenue_new"],
                f"{d_old_s} Total Revenue": fmt_old_day_value(r["Total_Revenue_old"], had_old),
                "Total Revenue 日环比": fmt_pct_change_display(
                    r["Total_Revenue_new"], r["Total_Revenue_old"], had_old, had_new
                ),
                f"{d_new_s} Total Profit": r["Total_Profit_new"],
                f"{d_old_s} Total Profit": fmt_old_day_value(r["Total_Profit_old"], had_old),
                "Total Profit 日环比": fmt_pct_change_display(
                    r["Total_Profit_new"], r["Total_Profit_old"], had_old, had_new, allow_negative=True
                ),
                f"{d_new_s} Total Profit Rate": fmt_pct(r["Total_Profit_Rate_new"], digits=2),
                f"{d_old_s} Total Profit Rate": fmt_old_day_value(
                    r["Total_Profit_Rate_old"], had_old, as_rate=True
                ),
                "Total Profit Rate 日环比": fmt_pct_change_display(
                    r["Total_Profit_Rate_new"],
                    r["Total_Profit_Rate_old"],
                    had_old,
                    had_new,
                    allow_negative=True,
                ),
                f"{d_new_s} reject比例": fmt_pct(r["reject_ratio_new"], digits=2),
                "reject比例 日环比": fmt_pct_change_display(
                    r["reject_ratio_new"], r["reject_ratio_old"], had_old, had_new
                ),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(f"{d_new_s} Total Revenue", ascending=False)
    return df


def build_declining_budget_table(sum_df: pd.DataFrame, date_new: date) -> pd.DataFrame:
    """近7天内最高利润日 vs 最近一天，利润下降超过5美金。"""
    keys = BUDGET_KEYS
    window_start = date_new - timedelta(days=6)
    window_df = sum_df[(sum_df["Time_date"] >= window_start) & (sum_df["Time_date"] <= date_new)]

    daily_budget = (
        window_df.groupby(keys + ["Time_date"], dropna=False)
        .agg(Total_Revenue=("Total Revenue", "sum"), Total_Profit=("Total Profit", "sum"))
        .reset_index()
    )
    daily_budget["Total_Profit_Rate"] = daily_budget.apply(
        lambda r: safe_div(r["Total_Profit"], r["Total_Revenue"], np.nan), axis=1
    )

    rows = []
    for budget_vals, grp in daily_budget.groupby(keys):
        if not isinstance(budget_vals, tuple):
            budget_vals = (budget_vals,)
        budget = dict(zip(keys, budget_vals))

        latest = grp[grp["Time_date"] == date_new]
        if latest.empty:
            continue
        latest_profit = latest["Total_Profit"].iloc[0]
        max_row = grp.loc[grp["Total_Profit"].idxmax()]
        max_profit = max_row["Total_Profit"]
        max_date = max_row["Time_date"]

        if latest_profit > max_profit - 5:
            continue

        profit_change = latest_profit - max_profit
        budget_row = pd.Series(budget)
        aff_text = analyze_affiliate_text(
            sum_df, keys, budget_row, max_date, date_new, profit_change
        )

        d_new_s = fmt_date(date_new)
        rows.append(
            {
                "source": budget["source"],
                "Advertiser": budget["Advertiser"],
                "Offer ID": budget["Offer ID"],
                "Adv Offer ID": budget["Adv Offer ID"],
                "App ID": budget["App ID"],
                "GEO": budget["GEO"],
                "Status": window_df[
                    (window_df["Time_date"] == date_new)
                    & (window_df["Offer ID"].astype(str) == str(budget["Offer ID"]))
                ]["Status"].iloc[0]
                if not window_df[
                    (window_df["Time_date"] == date_new)
                    & (window_df["Offer ID"].astype(str) == str(budget["Offer ID"]))
                ].empty
                else "",
                "利润最高日期": fmt_date(max_date),
                "利润最高日利润": max_profit,
                f"{d_new_s} Total Revenue": latest["Total_Revenue"].iloc[0],
                f"{d_new_s} Total Profit Rate": fmt_pct(latest["Total_Profit_Rate"].iloc[0], digits=2),
                f"{d_new_s} 利润": latest_profit,
                "利润变化幅度": profit_change,
                "分Affiliate影响因素分析": aff_text,
                "是否为新老预算": is_new_budget(sum_df, keys, budget_row, date_new),
            }
        )
    df = pd.DataFrame(rows)
    rev_col = f"{fmt_date(date_new)} Total Revenue"
    if not df.empty and rev_col in df.columns:
        df = df.sort_values(rev_col, ascending=False)
    return df


def write_excel(output_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)


def parse_date_arg(value: str | None) -> date | None:
    if not value:
        return None
    return pd.to_datetime(value).date()


def run(
    input_path: str | Path,
    output_path: str | Path | None = None,
    date_old: date | None = None,
    date_new: date | None = None,
) -> Path:
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "--分析结果.xlsx")
    else:
        output_path = Path(output_path)

    sum_df, event_df, reject_df = load_data(input_path)
    event_processed, reject_counts = process_event(event_df, reject_df)
    sum_processed = attach_reject_to_sum(sum_df, reject_counts)

    date_old, date_new = get_analysis_dates(sum_processed, date_old, date_new)
    daily = daily_summary(sum_processed, date_old, date_new)
    conclusion_sheet, meta = build_overall_conclusion_sheet(sum_processed, date_old, date_new)
    summary_text = conclusion_sheet.loc[0, "结论"]

    reject_summary = (
        event_processed.groupby(["Time_date", "Offer ID", "Advertiser", "Affiliate", "是否reject事件"], dropna=False)
        .size()
        .reset_index(name="事件数")
    )

    sheets = {
        "0-整体结论": conclusion_sheet,
        "1-按日汇总": daily,
        "2-近两日预算波动超5美金": build_budget_fluctuation_table(
            sum_processed, date_old, date_new
        ),
        "3-预算波动明细(含流量指标)": build_budget_detail_table(
            sum_processed, date_old, date_new
        ),
        "4-分广告主": build_advertiser_table(sum_processed, date_old, date_new),
        "5-分Affiliate": build_affiliate_table(sum_processed, date_old, date_new),
        "6-近7天利润下滑超5美金": build_declining_budget_table(sum_processed, date_new),
        "7-event_reject汇总": reject_summary,
        "8-sum_reject匹配汇总": sum_processed.groupby(
            ["Time_date", "Offer ID", "Advertiser", "Affiliate"], dropna=False
        )
        .agg(
            Total_Revenue=("Total Revenue", "sum"),
            Total_Profit=("Total Profit", "sum"),
            reject_count=("reject_count", "sum"),
        )
        .reset_index(),
    }

    write_excel(output_path, sheets)
    print(f"分析完成：{output_path}")
    print(f"分析日期：{fmt_date(date_old)} -> {fmt_date(date_new)}")
    print(f"结论：{str(summary_text)[:200]}...")
    print(f"预算分析行数：{len(conclusion_sheet) - 1}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="网盟日报分析")
    parser.add_argument(
        "-i",
        "--input",
        default="/Users/doraemonfang/Desktop/20260704--日报分析.xlsx",
        help="输入 Excel 路径",
    )
    parser.add_argument("-o", "--output", default=None, help="输出 Excel 路径")
    parser.add_argument(
        "--date-new",
        default=None,
        help="分析的新日期（不指定则取数据最新一天）",
    )
    parser.add_argument(
        "--date-old",
        default=None,
        help="分析的旧日期（不指定则取数据次新一天）",
    )
    args = parser.parse_args()
    run(
        args.input,
        args.output,
        date_old=parse_date_arg(args.date_old),
        date_new=parse_date_arg(args.date_new),
    )


if __name__ == "__main__":
    main()
