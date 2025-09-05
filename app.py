# app.py
import re
import os
import io
import sys
import time
import math
import requests
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
import numpy as np

import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx

import plotly.express as px

# Config
DATA_URL = "https://loyce.club/Merit/merit.all.txt"
CACHE_DIR = ".cache_merit"
PROCESSED_PARQUET = os.path.join(CACHE_DIR, "merit_processed.parquet")
RAW_CACHE = os.path.join(CACHE_DIR, "merit_raw.txt")
FORUM_TOPIC_BASE = "https://bitcointalk.org/index.php?topic="  # append topic id and .msg... if needed

os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="Bitcointalk Merit Dashboard", layout="wide")

# ---------------------
# Utilities and parsing
# ---------------------
@st.cache_data(show_spinner=False)
def download_raw(url: str, force: bool = False, timeout: int = 15) -> str:
    """
    Download raw text file with retries, save to RAW_CACHE and return content as string.
    If RAW_CACHE exists and not force, will reuse it.
    """
    if os.path.exists(RAW_CACHE) and not force:
        with open(RAW_CACHE, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    headers = {"User-Agent": "merit-dashboard/1.0 (+https://example.com)"}
    tries = 3
    for attempt in range(tries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            with open(RAW_CACHE, "w", encoding="utf-8") as f:
                f.write(text)
            return text
        except Exception as e:
            if attempt + 1 == tries:
                raise
            time.sleep(1 + attempt * 2)
    raise RuntimeError("Failed to download file")

# Example input line variants (unknown file format). We'll attempt flexible parsing:
# Some likely formats (examples):
# from:123 to:456 amount:1 date:2009-01-02 topic:789 board:45
# 123 -> 456 | 1 merit | 2009-01-02 | topic=789 board=45
# 123 456 1 2009-01-02 789 45
# Or CSV-like lines.
# We'll parse using regex to extract ID-like integers and dates and amounts.

date_patterns = [
    # yyyy-mm-dd or yyyy/mm/dd
    r"(\d{4}-\d{2}-\d{2})",
    r"(\d{4}/\d{2}/\d{2})",
    # dd-mon-yyyy or dd Mon yyyy
    r"(\d{1,2}\s+[A-Za-z]{3,}\s+\d{4})",
]

def try_parse_date(s: str):
    for pat in date_patterns:
        m = re.search(pat, s)
        if m:
            txt = m.group(1)
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
                try:
                    return datetime.strptime(txt, fmt)
                except Exception:
                    pass
    # fallback: try to parse any ISO-like token
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def parse_line(line: str):
    """
    Returns dict or None.
    Try to extract:
        from_id, to_id, amount (int), date (datetime), topic_id, board_id
    """
    s = line.strip()
    if not s:
        return None

    # quick tokenization (non-digit separators)
    # Extract all integers (sequences of digits). We'll then try to assign meaning by position/keywords.
    ints = re.findall(r"\d+", s)
    # Extract amounts: "merit" or "amount" nearby
    amount = None
    m = re.search(r"(?:amount|merit|merits)\s*[:=]?\s*(\d+)", s, flags=re.I)
    if m:
        amount = int(m.group(1))

    # from/to keywords
    from_id = None
    to_id = None
    m_from = re.search(r"(?:from|sender|by)\s*[:=]?\s*(\d+)", s, flags=re.I)
    m_to = re.search(r"(?:to|receiver)\s*[:=]?\s*(\d+)", s, flags=re.I)
    if m_from:
        from_id = int(m_from.group(1))
    if m_to:
        to_id = int(m_to.group(1))

    # topic/board keywords
    topic_id = None
    board_id = None
    m_topic = re.search(r"(?:topic|topicid|topic_id)\s*[:=]?\s*(\d+)", s, flags=re.I)
    m_board = re.search(r"(?:board|boardid|board_id)\s*[:=]?\s*(\d+)", s, flags=re.I)
    if m_topic:
        topic_id = int(m_topic.group(1))
    if m_board:
        board_id = int(m_board.group(1))

    # If keywords absent, heuristically map integers:
    # typical order: from to amount date topic board or from to amount topic board date
    if not (from_id and to_id):
        # if there are at least 2 ints, assume first=from second=to
        if len(ints) >= 2:
            if from_id is None:
                from_id = int(ints[0])
            if to_id is None:
                to_id = int(ints[1])

    # amount fallback
    if amount is None:
        # attempt to take the third integer as amount if sensible (small number)
        if len(ints) >= 3:
            cand = int(ints[2])
            # accept if not a year-like number
            if cand < 1000:
                amount = cand

    # topic/board fallback from later ints
    if topic_id is None and len(ints) >= 4:
        try:
            topic_id = int(ints[3])
        except:
            topic_id = None
    if board_id is None and len(ints) >= 5:
        try:
            board_id = int(ints[4])
        except:
            board_id = None

    # parse date
    date = try_parse_date(s)

    # Normalize
    if amount is None:
        amount = 1  # default single-merit if not specified

    return {
        "from_id": from_id,
        "to_id": to_id,
        "amount": int(amount),
        "date": date if not pd.isna(date) else pd.NaT,
        "topic_id": topic_id,
        "board_id": board_id,
        "raw": s,
    }

@st.cache_data(show_spinner=False)
def parse_text_to_df(raw_text: str, max_lines: int = None):
    """
    Parse the raw text file line by line into a DataFrame.
    If max_lines is set, only parse that many lines (useful for testing).
    """
    records = []
    # Try to iterate lines robustly
    for i, line in enumerate(io.StringIO(raw_text)):
        if max_lines and i >= max_lines:
            break
        parsed = parse_line(line)
        if parsed:
            records.append(parsed)
    if not records:
        return pd.DataFrame(
            columns=["from_id", "to_id", "amount", "date", "topic_id", "board_id", "raw"]
        )
    df = pd.DataFrame.from_records(records)
    # Normalize types
    df["from_id"] = pd.to_numeric(df["from_id"], errors="coerce").astype("Int64")
    df["to_id"] = pd.to_numeric(df["to_id"], errors="coerce").astype("Int64")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype(int)
    # date: ensure datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce").astype("Int64")
    df["board_id"] = pd.to_numeric(df["board_id"], errors="coerce").astype("Int64")
    # drop rows missing both from and to
    df = df.dropna(subset=["from_id", "to_id"], how="all").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_or_process(force_download=False, force_parse=False):
    """
    Returns processed DataFrame.
    Uses parquet cache if available and no force flags.
    """
    if os.path.exists(PROCESSED_PARQUET) and not (force_download or force_parse):
        try:
            df = pd.read_parquet(PROCESSED_PARQUET)
            return df
        except Exception:
            pass

    raw = download_raw(DATA_URL, force=force_download)
    df = parse_text_to_df(raw)
    # add derived columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M")
    # Save parquet for speed
    try:
        df.to_parquet(PROCESSED_PARQUET, index=False)
    except Exception:
        pass
    return df

# ---------------------
# Streamlit UI
# ---------------------
st.title("Bitcointalk — Merit Data Dashboard")

col1, col2 = st.columns([3, 1])
with col1:
    st.write("Source file:", DATA_URL)
with col2:
    st.button("Refresh data", key="refresh_button")

# Options in sidebar
st.sidebar.header("Controls")
force_download = st.sidebar.checkbox("Force download fresh dump", value=False)
force_parse = st.sidebar.checkbox("Force reparse", value=False)
sample_mode = st.sidebar.checkbox("Sample mode (first 50k lines)", value=False)
forum_base = st.sidebar.text_input("Forum topic base URL", value=FORUM_TOPIC_BASE)

st.sidebar.markdown("Hints: if the dashboard is slow, enable sample mode or increase server memory.")

# Load/process (show progress)
with st.spinner("Loading and parsing data — this may take a moment for a large dump..."):
    try:
        df = load_or_process(force_download=force_download, force_parse=force_parse)
    except Exception as e:
        st.error(f"Failed to load data from the URL: {e}")
        st.stop()

if df.empty:
    st.warning("No records parsed. The parser couldn't find recognizable merit entries.")
    st.write("Show a few raw lines to help debugging:")
    raw_preview = ""
    try:
        raw_preview = open(RAW_CACHE, "r", encoding="utf-8", errors="replace").read(10000)
    except Exception:
        raw_preview = "(no raw cache available)"
    st.code(raw_preview[:5000])
    st.stop()

# Basic stats
total_merits = int(df["amount"].sum())
total_entries = len(df)
unique_senders = int(df["from_id"].nunique(dropna=True))
unique_receivers = int(df["to_id"].nunique(dropna=True))
earliest = df["date"].min()
latest = df["date"].max()

st.metric("Total merit amount", f"{total_merits:,}")
st.metric("Total entries (lines)", f"{total_entries:,}", delta=None)
st.write(f"Unique senders: {unique_senders}, Unique receivers: {unique_receivers}")
st.write(f"Date range: {earliest} to {latest}")

# Sidebar filters
st.sidebar.markdown("### Filters")
min_date = st.sidebar.date_input("From date", value=earliest.date() if pd.notna(earliest) else None)
max_date = st.sidebar.date_input("To date", value=latest.date() if pd.notna(latest) else None)
user_search = st.sidebar.text_input("Search user id (exact) or partial username (not implemented)", value="")
topic_filter = st.sidebar.text_input("Topic ID (comma-separated)", value="")
board_filter = st.sidebar.text_input("Board ID (comma-separated)", value="")

# Apply date filters
filtered = df.copy()
if min_date:
    filtered = filtered[filtered["date"] >= pd.to_datetime(min_date)]
if max_date:
    filtered = filtered[filtered["date"] <= pd.to_datetime(max_date)]

# Apply topic/board filters if provided
def parse_id_list(s: str):
    if not s:
        return None
    parts = re.findall(r"\d+", s)
    return [int(x) for x in parts] if parts else None

topic_ids = parse_id_list(topic_filter)
board_ids = parse_id_list(board_filter)
if topic_ids:
    filtered = filtered[filtered["topic_id"].isin(topic_ids)]
if board_ids:
    filtered = filtered[filtered["board_id"].isin(board_ids)]

# Leaderboards
st.header("Leaderboards")
lcol1, lcol2, lcol3 = st.columns(3)

with lcol1:
    st.subheader("Top receivers (by total merits)")
    top_recv = (
        filtered.groupby("to_id", dropna=True)["amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"to_id": "user_id", "amount": "total_merits"})
    )
    st.dataframe(top_recv.head(50), height=350)

with lcol2:
    st.subheader("Top senders (by total merits given)")
    top_sent = (
        filtered.groupby("from_id", dropna=True)["amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"from_id": "user_id", "amount": "total_given"})
    )
    st.dataframe(top_sent.head(50), height=350)

with lcol3:
    st.subheader("Top topics (by total merits)")
    top_topics = (
        filtered.groupby("topic_id", dropna=True)["amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"topic_id": "topic_id", "amount": "total_merits"})
    )
    # Add link column if forum_base present
    if forum_base:
        top_topics["topic_link"] = top_topics["topic_id"].apply(
            lambda tid: urljoin(forum_base, str(tid)) if not pd.isna(tid) else ""
        )
    st.dataframe(top_topics.head(50), height=350)

# Charts
st.header("Charts")

# Merit over time (monthly)
time_df = (
    filtered.set_index("date")
    .resample("M")["amount"]
    .sum()
    .rename("monthly_merits")
    .reset_index()
)
if not time_df.empty:
    fig_time = px.line(time_df, x="date", y="monthly_merits", title="Merits per month")
    st.plotly_chart(fig_time, use_container_width=True)
else:
    st.write("No time-series data available for selected range.")

# Top 10 bar charts
st.subheader("Top 10 visualizations")
bcol1, bcol2 = st.columns(2)

with bcol1:
    top10_recv = top_recv.head(10)
    if not top10_recv.empty:
        fig_r = px.bar(top10_recv, x="user_id", y="total_merits", title="Top 10 receivers")
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.write("No top receivers to show.")

with bcol2:
    top10_topics = top_topics.head(10)
    if not top10_topics.empty:
        fig_t = px.bar(top10_topics, x="topic_id", y="total_merits", title="Top 10 topics")
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.write("No top topics to show.")

# Drill-down table and search by user id
st.header("Explore entries")
explore_user = st.text_input("Show entries for user id (from_id or to_id). Leave blank to show all.", value="")
if explore_user.strip().isdigit():
    uid = int(explore_user.strip())
    entries = filtered[(filtered["from_id"] == uid) | (filtered["to_id"] == uid)].sort_values(
        "date", ascending=False
    )
else:
    entries = filtered.sort_values("date", ascending=False).head(500)

# Add clickable topic links if possible (render as markdown)
def make_topic_link(tid):
    if pd.isna(tid):
        return ""
    return f"[{int(tid)}]({urljoin(forum_base, str(int(tid)))})" if forum_base else str(int(tid))

if not entries.empty:
    display_df = entries.copy()
    display_df["topic_link"] = display_df["topic_id"].apply(make_topic_link)
    display_df = display_df[
        ["date", "from_id", "to_id", "amount", "topic_id", "topic_link", "board_id", "raw"]
    ]
    st.dataframe(display_df.reset_index(drop=True), height=450)
else:
    st.write("No entries match the current filters.")

# Export options
st.sidebar.header("Export")
if st.sidebar.button("Download filtered as CSV"):
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download CSV", data=csv_bytes, file_name="merit_filtered.csv")

st.sidebar.markdown("App created: simple robust parser + Streamlit UI.")
st.sidebar.markdown("If the parser missed many entries, share a sample of the raw file for parser tuning.")
