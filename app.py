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
import traceback

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
    "https://bitcointalk.org/index.php?topic=5275032.0",
    "https://bitcointalk.org/index.php?topic=5275032.1980",
    "https://bitcointalk.org/index.php?topic=5275032.2000",
    "https://bitcointalk.org/index.php?topic=5275032.2020"
]
LOCAL_SOURCE_THREAD = "https://bitcointalk.org/index.php?topic=4428616.0"

# Data caching functions
@st.cache_data(ttl=3600, show_spinner="Downloading and parsing merit data...")
def download_and_parse_merit_data():
    """Download and parse the merit data with robust schema detection"""
    try:
        response = requests.get(MERIT_URL, timeout=30)
        response.raise_for_status()
        
        # Decompress the xz file
        with lzma.open(BytesIO(response.content)) as f:
            decompressed_data = f.read()
        
        raw_data = decompressed_data.decode('utf-8')
        lines = raw_data.strip().split('\n')
        
        # Try to detect schema from first few lines
        schema = None
        if lines and ';' in lines[0]:
            # Use the first line as header
            schema = [col.strip() for col in lines[0].split(';')]
            data_lines = lines[1:]
        else:
            # Fallback to default schema
            schema = ['timestamp', 'amount', 'from_user', 'to_user', 'post_url']
            data_lines = lines
        
        # Parse data lines
        transactions = []
        for line in data_lines:
            if not line.strip():
                continue
                
            if ';' in line:
                parts = [part.strip() for part in line.split(';')]
            else:
                # Fallback: extract using regex patterns
                date_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
                amount_match = re.search(r'(\d+)\s*sMerit', line, re.IGNORECASE)
                users_match = re.search(r'from\s+([^\s]+)\s+to\s+([^\s]+)', line, re.IGNORECASE)
                url_match = re.search(r'https?://[^\s]+', line)
                
                parts = [
                    date_match.group(0) if date_match else '',
                    amount_match.group(1) if amount_match else '0',
                    users_match.group(1) if users_match else '',
                    users_match.group(2) if users_match else '',
                    url_match.group(0) if url_match else ''
                ]
            
            # Ensure we have exactly 5 parts
            if len(parts) < 5:
                parts.extend([''] * (5 - len(parts)))
            elif len(parts) > 5:
                parts = parts[:5]
            
            transactions.append(parts)
        
        df = pd.DataFrame(transactions, columns=schema)
        
        # Convert timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Convert amount to numeric
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        # Clean user columns
        for col in ['from_user', 'to_user']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df, schema
    
    except Exception as e:
        st.error(f"Error downloading merit data: {str(e)}")
        return pd.DataFrame(), []

@st.cache_data(ttl=3600, show_spinner="Scraping candidate merit sources...")
def scrape_candidate_sources():
    """Scrape community threads for potential merit sources"""
    candidate_sources = set()
    
    # Scrape observation threads
    for url in OBSERVATION_THREADS:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for posts containing source mentions
            posts = soup.find_all('div', class_='post')
            for post in posts:
                text = post.get_text().lower()
                if any(phrase in text for phrase in ['merit source', 'new source', 'accepted', 'source']):
                    # Extract usernames (simple heuristic)
                    username_matches = re.findall(r'@?([a-zA-Z0-9_]{3,})', text)
                    candidate_sources.update(username_matches)
            
            time.sleep(2)  # Be polite to the server
        except Exception as e:
            st.warning(f"Could not scrape {url}: {str(e)}")
    
    # Scrape local source thread
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(LOCAL_SOURCE_THREAD, timeout=15, headers=headers)
        response.raise_for_status()
        
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
                # Also look for mentions of sources
                if 'source' in line.lower():
                    username_matches = re.findall(r'@?([a-zA-Z0-9_]{3,})', line)
                    candidate_sources.update(username_matches)
        
        time.sleep(2)
    except Exception as e:
        st.warning(f"Could not scrape local source thread: {str(e)}")
    
    return list(candidate_sources)

# Inference functions
def calculate_merit_metrics(merit_df, candidate_sources):
    """Calculate rolling metrics and infer merit sources"""
    if merit_df.empty:
        return pd.DataFrame()
    
    try:
        # Ensure we have the necessary columns
        if 'from_user' not in merit_df.columns or 'amount' not in merit_df.columns or 'timestamp' not in merit_df.columns:
            st.error("Merit data doesn't contain required columns")
            return pd.DataFrame()
        
        # Filter out system users and invalid data
        df_filtered = merit_df[
            (merit_df['from_user'].str.len() > 0) & 
            (merit_df['from_user'] != 'System') &
            (merit_df['amount'] > 0)
        ].copy()
        
        if df_filtered.empty:
            return pd.DataFrame()
        
        # Create a date column without time
        df_filtered['date'] = pd.to_datetime(df_filtered['timestamp']).dt.date
        
        # Group by sender and date
        daily_sent = df_filtered.groupby(['from_user', 'date'])['amount'].sum().reset_index()
        
        # Create pivot table for rolling calculations
        pivot = daily_sent.pivot_table(
            index='date', 
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
            if user_data.sum() == 0:
                continue
                
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
                # Only include users with some activity
                if user_data.sum() > 0:
                    confidence = "Low"
                else:
                    continue
            
            user_metrics.append({
                'username': user,
                'total_sent': user_data.sum(),
                'max_30d_sent': rolling_max,
                'median_30d_sent': rolling_median,
                'in_scraped_list': in_scraped_list,
                'confidence': confidence
            })
        
        return pd.DataFrame(user_metrics)
    
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return pd.DataFrame()

# UI Components
def render_sidebar(merit_df, schema):
    """Render the sidebar with filters and info"""
    st.sidebar.title("MeritSource Reader")
    
    st.sidebar.subheader("Data Schema")
    if schema:
        st.sidebar.write(", ".join(schema))
    else:
        st.sidebar.write("No schema detected")
    
    st.sidebar.subheader("Filters")
    min_merit = st.sidebar.slider("Minimum sMerit", 0, 500, 10)
    
    confidence_options = ["High", "Medium", "Low"]
    confidence_filter = st.sidebar.multiselect(
        "Confidence Level",
        options=confidence_options,
        default=confidence_options
    )
    
    return min_merit, confidence_filter

def render_kpi_cards(merit_df):
    """Render KPI cards on home tab"""
    if merit_df.empty:
        st.warning("No data available for KPIs")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", len(merit_df))
    
    with col2:
        unique_senders = merit_df['from_user'].nunique() if 'from_user' in merit_df.columns else 0
        st.metric("Unique Senders", unique_senders)
    
    with col3:
        unique_receivers = merit_df['to_user'].nunique() if 'to_user' in merit_df.columns else 0
        st.metric("Unique Receivers", unique_receivers)
    
    # Time-based metrics
    if 'timestamp' in merit_df.columns and 'amount' in merit_df.columns:
        now = datetime.now()
        last_7_days = now - timedelta(days=7)
        last_30_days = now - timedelta(days=30)
        
        recent_7d = merit_df[merit_df['timestamp'] >= last_7_days] if not merit_df.empty else pd.DataFrame()
        recent_30d = merit_df[merit_df['timestamp'] >= last_30_days] if not merit_df.empty else pd.DataFrame()
        
        col4, col5, col6 = st.columns(3)
        with col4:
            sent_7d = recent_7d['amount'].sum() if not recent_7d.empty else 0
            st.metric("sMerit Sent (7d)", sent_7d)
        with col5:
            sent_30d = recent_30d['amount'].sum() if not recent_30d.empty else 0
            st.metric("sMerit Sent (30d)", sent_30d)
        with col6:
            total_sent = merit_df['amount'].sum() if not merit_df.empty else 0
            st.metric("Total sMerit", total_sent)

def render_merit_sources_tab(metrics_df, min_merit, confidence_filter):
    """Render the merit sources tab"""
    st.header("Inferred Merit Sources")
    
    if metrics_df.empty:
        st.info("No merit source metrics calculated yet. Refresh data to generate metrics.")
        return
    
    # Apply filters
    filtered_df = metrics_df[
        (metrics_df['max_30d_sent'] >= min_merit) &
        (metrics_df['confidence'].isin(confidence_filter))
    ]
    
    if filtered_df.empty:
        st.info("No sources match the current filters.")
        return
    
    # Display metrics
    st.dataframe(
        filtered_df.sort_values('max_30d_sent', ascending=False),
        column_config={
            "username": "Username",
            "total_sent": st.column_config.NumberColumn("Total Sent"),
            "max_30d_sent": st.column_config.NumberColumn("Max 30d Sent"),
            "median_30d_sent": st.column_config.NumberColumn("Median 30d Sent"),
            "in_scraped_list": "In Scraped List",
            "confidence": "Confidence"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Export option
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        "Export CSV",
        data=csv,
        file_name="merit_sources.csv",
        mime="text/csv",
        key="export_sources"
    )

def render_search_tab(merit_df):
    """Render the search functionality"""
    st.header("Search Merit Transactions")
    
    if merit_df.empty:
        st.info("No data available for search. Please refresh data first.")
        return
    
    search_term = st.text_input("Search by username, post URL, or other terms")
    
    if search_term:
        # Search across all columns
        mask = np.column_stack([merit_df[col].astype(str).str.contains(search_term, case=False, na=False) 
                              for col in merit_df.columns])
        search_results = merit_df.loc[mask.any(axis=1)]
        
        if search_results.empty:
            st.info("No results found for your search term.")
        else:
            st.write(f"Found {len(search_results)} matching transactions:")
            st.dataframe(search_results.head(50), use_container_width=True)
            
            # Export search results
            csv = search_results.to_csv(index=False)
            st.download_button(
                "Export Search Results",
                data=csv,
                file_name="merit_search_results.csv",
                mime="text/csv",
                key="export_search"
            )
    else:
        st.info("Enter a search term to find matching transactions")

def main():
    """Main application function"""
    st.title("⭐ MeritSource Reader for Bitcointalk")
    
    # Initialize session state for data
    if 'merit_df' not in st.session_state:
        st.session_state.merit_df = pd.DataFrame()
    if 'schema' not in st.session_state:
        st.session_state.schema = []
    if 'candidate_sources' not in st.session_state:
        st.session_state.candidate_sources = []
    if 'metrics_df' not in st.session_state:
        st.session_state.metrics_df = pd.DataFrame()
    
    # Download data button
    if st.button("🔄 Refresh All Data", type="primary"):
        with st.spinner("Refreshing all data..."):
            # Clear cache and reload
            st.cache_data.clear()
            st.session_state.merit_df, st.session_state.schema = download_and_parse_merit_data()
            st.session_state.candidate_sources = scrape_candidate_sources()
            st.session_state.metrics_df = calculate_merit_metrics(
                st.session_state.merit_df, st.session_state.candidate_sources
            )
        st.success("Data refreshed successfully!")
    
    # Load data if not already loaded
    if st.session_state.merit_df.empty:
        with st.spinner("Loading data for the first time..."):
            st.session_state.merit_df, st.session_state.schema = download_and_parse_merit_data()
            st.session_state.candidate_sources = scrape_candidate_sources()
            st.session_state.metrics_df = calculate_merit_metrics(
                st.session_state.merit_df, st.session_state.candidate_sources
            )
    
    # Sidebar
    min_merit, confidence_filter = render_sidebar(st.session_state.merit_df, st.session_state.schema)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "⭐ Merit Sources", "🔍 Search"])
    
    with tab1:
        st.header("Merit Activity Overview")
        
        if st.session_state.merit_df.empty:
            st.warning("No merit data available. Please refresh or check connection.")
        else:
            render_kpi_cards(st.session_state.merit_df)
            
            # Show data preview
            st.subheader("Recent Transactions")
            st.dataframe(st.session_state.merit_df.head(10), use_container_width=True)
            
            # Data info
            st.subheader("Data Information")
            st.write(f"Data loaded: {len(st.session_state.merit_df)} transactions")
            st.write(f"Time range: {st.session_state.merit_df['timestamp'].min()} to {st.session_state.merit_df['timestamp'].max()}" 
                    if 'timestamp' in st.session_state.merit_df.columns and not st.session_state.merit_df.empty else "Unknown")
    
    with tab2:
        render_merit_sources_tab(st.session_state.metrics_df, min_merit, confidence_filter)
    
    with tab3:
        render_search_tab(st.session_state.merit_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **About the data:** 
        - Merit feed covers approximately 120 days of recent activity from [bitcointalk.org/merit.txt.xz](https://bitcointalk.org/merit.txt.xz)
        - Older history accumulates locally but isn't guaranteed to persist on free hosting
        - Merit source detection uses heuristics and community scraping (not official)
        - Data refreshes automatically every hour, or manually with the Refresh button
        """
    )

if __name__ == "__main__":
    main()
