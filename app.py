# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import time
import re

# Set page configuration
st.set_page_config(
    page_title="MeritPulse - Bitcointalk Merit Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
    }
    .source-table {
        font-size: 0.8rem;
    }
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">MeritPulse</h1>', unsafe_allow_html=True)
st.markdown("### Track and analyze Bitcointalk merit source activity in real-time")

# Function to fetch data with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to parse merit data
def parse_merit_data(data_text):
    lines = data_text.strip().split('\n')
    merit_data = []
    
    for line in lines:
        if line.startswith('#'):
            continue
            
        parts = line.split('\t')
        if len(parts) >= 4:
            try:
                date_str = parts[0].strip()
                giver_id = parts[1].strip()
                receiver_id = parts[2].strip()
                amount = int(parts[3].strip())
                
                # Parse date (assuming format: YYYY-MM-DD HH:MM:SS)
                date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                
                merit_data.append({
                    'date': date,
                    'giver_id': giver_id,
                    'receiver_id': receiver_id,
                    'amount': amount
                })
            except (ValueError, IndexError):
                continue
                
    return pd.DataFrame(merit_data)

# Function to parse merit sources
def parse_merit_sources(source_text):
    lines = source_text.strip().split('\n')
    sources = []
    
    for line in lines:
        if line.startswith('#'):
            continue
            
        parts = line.split('\t')
        if len(parts) >= 2:
            try:
                user_id = parts[0].strip()
                username = parts[1].strip()
                sources.append({
                    'user_id': user_id,
                    'username': username
                })
            except (ValueError, IndexError):
                continue
                
    return pd.DataFrame(sources)

# Load data function
def load_data():
    with st.spinner('Fetching latest merit data...'):
        # Fetch merit data
        merit_text = fetch_data("https://loyce.club/merit/all_merit.txt")
        if merit_text is None:
            st.error("Failed to fetch merit data. Using sample data instead.")
            # For demo purposes, create sample data if fetch fails
            return create_sample_data()
        
        # Fetch merit sources
        sources_text = fetch_data("https://loyce.club/merit/merit_sources.txt")
        if sources_text is None:
            st.error("Failed to fetch merit sources. Using sample data instead.")
            # For demo purposes, create sample data if fetch fails
            return create_sample_data()
    
    # Parse data
    merit_df = parse_merit_data(merit_text)
    sources_df = parse_merit_sources(sources_text)
    
    # Merge to get usernames
    merit_df = merit_df.merge(sources_df, left_on='giver_id', right_on='user_id', how='left')
    merit_df.rename(columns={'username': 'giver_username'}, inplace=True)
    
    merit_df = merit_df.merge(sources_df, left_on='receiver_id', right_on='user_id', how='left')
    merit_df.rename(columns={'username': 'receiver_username'}, inplace=True)
    
    # Fill NaN usernames with user IDs
    merit_df['giver_username'] = merit_df['giver_username'].fillna(merit_df['giver_id'])
    merit_df['receiver_username'] = merit_df['receiver_username'].fillna(merit_df['receiver_id'])
    
    return merit_df, sources_df

# Function to create sample data for demo purposes
def create_sample_data():
    # Create sample merit data
    dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='H')
    sample_size = min(5000, len(dates))
    
    merit_data = []
    for i in range(sample_size):
        date = dates[i]
        giver_id = f"user_{np.random.randint(1, 50)}"
        receiver_id = f"user_{np.random.randint(51, 150)}"
        amount = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.3, 0.1, 0.07, 0.03])
        
        merit_data.append({
            'date': date,
            'giver_id': giver_id,
            'receiver_id': receiver_id,
            'amount': amount
        })
    
    merit_df = pd.DataFrame(merit_data)
    
    # Create sample sources
    sources_data = []
    for i in range(1, 151):
        sources_data.append({
            'user_id': f"user_{i}",
            'username': f"SampleUser{i}"
        })
    
    sources_df = pd.DataFrame(sources_data)
    
    # Merge to get usernames
    merit_df = merit_df.merge(sources_df, left_on='giver_id', right_on='user_id', how='left')
    merit_df.rename(columns={'username': 'giver_username'}, inplace=True)
    
    merit_df = merit_df.merge(sources_df, left_on='receiver_id', right_on='user_id', how='left')
    merit_df.rename(columns={'username': 'receiver_username'}, inplace=True)
    
    return merit_df, sources_df

# Calculate metrics and prepare data for visualization
def prepare_visualization_data(merit_df):
    # Calculate date 30 days ago
    thirty_days_ago = datetime.now() - timedelta(days=30)
    
    # Filter data for last 30 days
    recent_merit_df = merit_df[merit_df['date'] >= thirty_days_ago]
    
    # Top merit sources (all time)
    top_sources_alltime = merit_df.groupby(['giver_id', 'giver_username']).agg({
        'amount': ['sum', 'count']
    }).reset_index()
    top_sources_alltime.columns = ['giver_id', 'giver_username', 'total_merit', 'total_transactions']
    top_sources_alltime = top_sources_alltime.sort_values('total_merit', ascending=False)
    
    # Top merit sources (last 30 days)
    top_sources_recent = recent_merit_df.groupby(['giver_id', 'giver_username']).agg({
        'amount': ['sum', 'count']
    }).reset_index()
    top_sources_recent.columns = ['giver_id', 'giver_username', 'recent_merit', 'recent_transactions']
    top_sources_recent = top_sources_recent.sort_values('recent_merit', ascending=False)
    
    # Timeline data for last 30 days
    timeline_data = recent_merit_df.groupby(recent_merit_df['date'].dt.date).agg({
        'amount': 'sum',
        'giver_id': 'count'
    }).reset_index()
    timeline_data.columns = ['date', 'total_merit', 'transaction_count']
    
    return top_sources_alltime, top_sources_recent, timeline_data

# Create visualizations
def create_visualizations(timeline_data, top_sources_alltime, top_sources_recent):
    # Create merit timeline chart
    fig_timeline = px.area(
        timeline_data, 
        x='date', 
        y='total_merit',
        title='Merit Given Over Time (Last 30 Days)',
        labels={'date': 'Date', 'total_merit': 'Total Merit'}
    )
    fig_timeline.update_layout(
        xaxis=dict(rangeslider=dict(visible=True)),
        height=400
    )
    
    # Create top sources chart (all time)
    fig_top_alltime = px.bar(
        top_sources_alltime.head(10),
        x='giver_username',
        y='total_merit',
        title='Top 10 Merit Sources (All Time)',
        labels={'giver_username': 'Username', 'total_merit': 'Total Merit Given'}
    )
    fig_top_alltime.update_layout(height=400)
    
    # Create top sources chart (last 30 days)
    fig_top_recent = px.bar(
        top_sources_recent.head(10),
        x='giver_username',
        y='recent_merit',
        title='Top 10 Merit Sources (Last 30 Days)',
        labels={'giver_username': 'Username', 'recent_merit': 'Merit Given'}
    )
    fig_top_recent.update_layout(height=400)
    
    return fig_timeline, fig_top_alltime, fig_top_recent

# Display metrics
def display_metrics(merit_df, recent_merit_df):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_merit = merit_df['amount'].sum()
        st.metric("Total Merit Given", f"{total_merit:,}")
    
    with col2:
        total_transactions = len(merit_df)
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col3:
        recent_merit = recent_merit_df['amount'].sum()
        st.metric("Merit Given (30 Days)", f"{recent_merit:,}")
    
    with col4:
        recent_transactions = len(recent_merit_df)
        st.metric("Transactions (30 Days)", f"{recent_transactions:,}")

# User search functionality
def user_search(merit_df, sources_df):
    st.markdown("---")
    st.markdown('<div class="subheader">Search Merit Source Activity</div>', unsafe_allow_html=True)
    
    # Create search options
    search_col1, search_col2 = st.columns([2, 1])
    
    with search_col1:
        search_options = ['By Username', 'By User ID']
        search_method = st.radio("Search by:", search_options, horizontal=True)
    
    with search_col2:
        if search_method == 'By Username':
            user_list = sources_df['username'].tolist()
            search_query = st.selectbox("Select username", user_list)
        else:
            user_list = sources_df['user_id'].tolist()
            search_query = st.selectbox("Select user ID", user_list)
    
    if search_query:
        if search_method == 'By Username':
            user_data = sources_df[sources_df['username'] == search_query]
            if not user_data.empty:
                user_id = user_data['user_id'].iloc[0]
            else:
                st.warning("User not found")
                return
        else:
            user_id = search_query
            user_data = sources_df[sources_df['user_id'] == user_id]
            if not user_data.empty:
                search_query = user_data['username'].iloc[0]
            else:
                st.warning("User ID not found")
                return
        
        # Filter merit data for this user
        user_merit_given = merit_df[merit_df['giver_id'] == user_id]
        user_merit_received = merit_df[merit_df['receiver_id'] == user_id]
        
        if user_merit_given.empty and user_merit_received.empty:
            st.info(f"No merit activity found for {search_query} ({user_id})")
            return
        
        # Display user metrics
        st.markdown(f"### Merit Activity for {search_query} ({user_id})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_given = user_merit_given['amount'].sum()
            st.metric("Total Merit Given", f"{total_given:,}")
        
        with col2:
            transactions_given = len(user_merit_given)
            st.metric("Transactions Given", f"{transactions_given:,}")
        
        with col3:
            total_received = user_merit_received['amount'].sum()
            st.metric("Total Merit Received", f"{total_received:,}")
        
        with col4:
            transactions_received = len(user_merit_received)
            st.metric("Transactions Received", f"{transactions_received:,}")
        
        # Display recent activity
        st.subheader("Recent Merit Given")
        if not user_merit_given.empty:
            recent_given = user_merit_given.sort_values('date', ascending=False).head(10)
            st.dataframe(recent_given[['date', 'receiver_username', 'amount']], 
                         column_config={
                             "date": "Date",
                             "receiver_username": "Receiver",
                             "amount": "Amount"
                         })
        else:
            st.info("No merit given activity found")
        
        st.subheader("Recent Merit Received")
        if not user_merit_received.empty:
            recent_received = user_merit_received.sort_values('date', ascending=False).head(10)
            st.dataframe(recent_received[['date', 'giver_username', 'amount']], 
                         column_config={
                             "date": "Date",
                             "giver_username": "Giver",
                             "amount": "Amount"
                         })
        else:
            st.info("No merit received activity found")

# Main app function
def main():
    # Load data
    merit_df, sources_df = load_data()
    
    # Calculate 30 days ago for filtering
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_merit_df = merit_df[merit_df['date'] >= thirty_days_ago]
    
    # Display metrics
    display_metrics(merit_df, recent_merit_df)
    
    # Prepare visualization data
    top_sources_alltime, top_sources_recent, timeline_data = prepare_visualization_data(merit_df)
    
    # Create visualizations
    fig_timeline, fig_top_alltime, fig_top_recent = create_visualizations(
        timeline_data, top_sources_alltime, top_sources_recent
    )
    
    # Display charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_timeline, use_container_width=True)
        st.plotly_chart(fig_top_alltime, use_container_width=True)
    
    with col2:
        # Display data tables
        st.markdown('<div class="subheader">Top Merit Sources (All Time)</div>', unsafe_allow_html=True)
        st.dataframe(
            top_sources_alltime.head(20)[['giver_username', 'total_merit', 'total_transactions']],
            column_config={
                "giver_username": "Username",
                "total_merit": "Total Merit",
                "total_transactions": "Transactions"
            },
            height=400
        )
        
        st.markdown('<div class="subheader">Top Merit Sources (Last 30 Days)</div>', unsafe_allow_html=True)
        st.dataframe(
            top_sources_recent.head(20)[['giver_username', 'recent_merit', 'recent_transactions']],
            column_config={
                "giver_username": "Username",
                "recent_merit": "Merit Given",
                "recent_transactions": "Transactions"
            },
            height=400
        )
    
    st.plotly_chart(fig_top_recent, use_container_width=True)
    
    # User search functionality
    user_search(merit_df, sources_df)
    
    # Data export functionality
    st.markdown("---")
    st.markdown('<div class="subheader">Export Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Top Sources (All Time) to CSV"):
            csv = top_sources_alltime.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="top_merit_sources_all_time.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Top Sources (30 Days) to CSV"):
            csv = top_sources_recent.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="top_merit_sources_30_days.csv",
                mime="text/csv"
            )
    
    # Footer with last update time
    st.markdown("---")
    st.caption(f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
               "Data source: https://loyce.club/merit/")

# Run the app
if __name__ == "__main__":
    main()
