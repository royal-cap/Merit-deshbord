# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import lzma
import json
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import time

# Set page configuration
st.set_page_config(
    page_title="MeritSource Reader",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MERIT_URL = "https://bitcointalk.org/merit.txt.xz"
OBSERVATION_THREADS = [
    "https://bitcointalk.org/index.php?topic=5275032.0",  # First page
    "https://bitcointalk.org/index.php?topic=5275032.1980",  # Recent pages
    "https://bitcointalk.org/index.php?topic=5275032.2000",
    "https://bitcointalk.org/index.php?topic=5275032.2020"
]
LOCAL_SOURCE_THREAD = "https://bitcointalk.org/index.php?topic=4428616.0"

# Data caching functions
@st.cache_data(ttl=3600, show_spinner=False)
def download_and_parse_merit_data():
    """Download and parse the merit data with robust schema detection"""
    try:
        response = requests.get(MERIT_URL, timeout=30)
        response.raise_for_status()
        
        with lzma.open(BytesIO(response.content)) as f:
            raw_data = f.read().decode('utf-8')
        
        lines = raw_data.strip().split('\n')
        
        # Try to detect schema from first few lines
        schema = None
        if lines and ';' in lines[0]:
            schema = lines[0].split(';')
            data_lines = lines[1:]
        else:
            # Fallback to regex parsing if no clear schema
            data_lines = lines
            schema = ['timestamp', 'amount', 'from_user', 'to_user', 'post_url']
        
        # Parse data lines
        transactions = []
        for line in data_lines:
            if ';' in line:
                parts = line.split(';')
            else:
                # Fallback: extract datetime patterns and URLs
                date_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
                url_match = re.search(r'https?://[^\s]+', line)
                amount_match = re.search(r'(\d+)\s*sMerit', line)
                users_match = re.findall(r'from\s+(\w+)\s+to\s+(\w+)', line)
                
                if date_match and users_match:
                    parts = [
                        date_match.group(0),
                        amount_match.group(1) if amount_match else '',
                        users_match[0][0] if users_match else '',
                        users_match[0][1] if users_match else '',
                        url_match.group(0) if url_match else ''
                    ]
                else:
                    continue
            
            if len(parts) >= 5:
                transactions.append(parts[:5])  # Take first 5 fields only
        
        df = pd.DataFrame(transactions, columns=schema[:5])
        
        # Convert timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Convert amount to numeric
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        return df, schema
    
    except Exception as e:
        st.error(f"Error downloading merit data: {str(e)}")
        return pd.DataFrame(), []

@st.cache_data(ttl=3600, show_spinner=False)
def scrape_candidate_sources():
    """Scrape community threads for potential merit sources"""
    candidate_sources = set()
    
    # Scrape observation threads
    for url in OBSERVATION_THREADS:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for posts containing source mentions
            posts = soup.find_all('div', class_='post')
            for post in posts:
                text = post.get_text().lower()
                if any(phrase in text for phrase in ['merit source', 'new source', 'accepted']):
                    # Extract usernames (simple heuristic)
                    username_matches = re.findall(r'@?(\w{3,})', text)
                    candidate_sources.update(username_matches)
            
            time.sleep(1)  # Be polite
        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
    
    # Scrape local source thread
    try:
        response = requests.get(LOCAL_SOURCE_THREAD, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        posts = soup.find_all('div', class_='post')
        
        for post in posts:
            text = post.get_text()
            # Look for lines with username patterns
            lines = text.split('\n')
            for line in lines:
                if re.match(r'^\w+:\s*[0-9,]+\s*$', line.strip()):
                    username = line.split(':')[0].strip()
                    candidate_sources.add(username)
        
        time.sleep(1)
    except Exception as e:
        st.error(f"Error scraping local source thread: {str(e)}")
    
    return list(candidate_sources)

# Inference functions
def calculate_merit_metrics(df, candidate_sources):
    """Calculate rolling metrics and infer merit sources"""
    if df.empty:
        return pd.DataFrame()
    
    # Group by sender and date
    df_sent = df.groupby([df['from_user'], pd.Grouper(key='timestamp', freq='D')])['amount'].sum().reset_index()
    
    # Create pivot table for rolling calculations
    pivot = df_sent.pivot_table(
        index='timestamp', 
        columns='from_user', 
        values='amount', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate rolling 30-day sums
    rolling_30d = pivot.rolling(window=30, min_periods=1).sum()
    
    # Calculate metrics per user
    user_metrics = []
    for user in pivot.columns:
        user_data = pivot[user]
        rolling_max = rolling_30d[user].max()
        rolling_median = rolling_30d[user].median()
        
        # Check criteria for merit source
        meets_volume = rolling_max >= 150 or rolling_median >= 120
        in_scraped_list = user in candidate_sources
        
        # Determine confidence
        if meets_volume and in_scraped_list:
            confidence = "High"
        elif meets_volume or in_scraped_list:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        user_metrics.append({
            'username': user,
            'max_30d_sent': rolling_max,
            'median_30d_sent': rolling_median,
            'in_scraped_list': in_scraped_list,
            'confidence': confidence
        })
    
    return pd.DataFrame(user_metrics)

# UI Components
def render_sidebar(merit_df, schema):
    """Render the sidebar with filters and info"""
    st.sidebar.title("MeritSource Reader")
    st.sidebar.subheader("Data Schema")
    st.sidebar.json(schema)
    
    st.sidebar.subheader("Filters")
    min_merit = st.sidebar.slider("Minimum sMerit", 0, 500, 10)
    confidence_filter = st.sidebar.multiselect(
        "Confidence Level",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )
    
    return min_merit, confidence_filter

def render_kpi_cards(merit_df):
    """Render KPI cards on home tab"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", len(merit_df))
    
    with col2:
        unique_senders = merit_df['from_user'].nunique()
        st.metric("Unique Senders", unique_senders)
    
    with col3:
        unique_receivers = merit_df['to_user'].nunique()
        st.metric("Unique Receivers", unique_receivers)
    
    # Time-based metrics
    now = datetime.now()
    last_7_days = now - timedelta(days=7)
    last_30_days = now - timedelta(days=30)
    
    recent_7d = merit_df[merit_df['timestamp'] >= last_7_days]
    recent_30d = merit_df[merit_df['timestamp'] >= last_30_days]
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("sMerit Sent (7d)", recent_7d['amount'].sum())
    with col5:
        st.metric("sMerit Sent (30d)", recent_30d['amount'].sum())
    with col6:
        st.metric("Total sMerit", merit_df['amount'].sum())

def render_merit_sources_tab(metrics_df, min_merit, confidence_filter):
    """Render the merit sources tab"""
    st.header("Inferred Merit Sources")
    
    # Apply filters
    filtered_df = metrics_df[
        (metrics_df['max_30d_sent'] >= min_merit) &
        (metrics_df['confidence'].isin(confidence_filter))
    ]
    
    # Display metrics
    st.dataframe(
        filtered_df,
        column_config={
            "username": "Username",
            "max_30d_sent": st.column_config.NumberColumn("Max 30d Sent"),
            "median_30d_sent": st.column_config.NumberColumn("Median 30d Sent"),
            "in_scraped_list": "In Scraped List",
            "confidence": "Confidence"
        },
        hide_index=True
    )
    
    # Export option
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        "Export CSV",
        data=csv,
        file_name="merit_sources.csv",
        mime="text/csv"
    )

def main():
    """Main application function"""
    st.title("MeritSource Reader for Bitcointalk")
    
    # Download data
    with st.spinner("Downloading and parsing merit data..."):
        merit_df, schema = download_and_parse_merit_data()
    
    with st.spinner("Scraping candidate merit sources..."):
        candidate_sources = scrape_candidate_sources()
    
    # Calculate metrics
    metrics_df = calculate_merit_metrics(merit_df, candidate_sources)
    
    # Sidebar
    min_merit, confidence_filter = render_sidebar(merit_df, schema)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Merit Sources", "Boards & Topics"])
    
    with tab1:
        st.header("Merit Activity Overview")
        
        if merit_df.empty:
            st.warning("No merit data available. Please refresh or check connection.")
        else:
            render_kpi_cards(merit_df)
            
            # Global search
            st.subheader("Global Search")
            search_term = st.text_input("Search by username, topic, or URL")
            
            if search_term:
                search_results = merit_df[
                    merit_df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)
                ]
                st.dataframe(search_results.head(20))
            
            # Refresh button
            if st.button("Refresh Data"):
                st.cache_data.clear()
                st.experimental_rerun()
    
    with tab2:
        if metrics_df.empty:
            st.warning("No metrics calculated. Need merit data first.")
        else:
            render_merit_sources_tab(metrics_df, min_merit, confidence_filter)
    
    with tab3:
        st.header("Boards & Topics Analysis")
        st.info("This feature will be implemented in a future version.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **About the data:** 
        - Merit feed covers approximately 120 days of recent activity
        - Older history accumulates locally but isn't guaranteed to persist
        - Merit source detection uses heuristics and community scraping
        - [Official merit feed](https://bitcointalk.org/merit.txt.xz)
        """
    )

if __name__ == "__main__":
    main()
