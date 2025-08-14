import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from datetime import datetime
from src.utils import load_feedback_data, get_env_var
from src.rag.rag_answerer import generate_rag_answer, load_vector_store
from src.analysis.trend_detection import detect_trends, generate_trend_visualizations
from src.analysis.cohort_analysis import perform_cohort_analysis, generate_cohort_visualizations

# Set page config
st.set_page_config(page_title="Customer Feedback Analysis", page_icon="📊", layout="wide")

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
RAG_DIR = os.path.join(DATA_DIR, 'rag_output')
VECTOR_STORE_DIR = os.path.join(RAG_DIR, 'vector_store')
INSIGHTS_PATH = os.path.join(RAG_DIR, 'topic_insights.json')
EVAL_RESULTS_PATH = os.path.join(RAG_DIR, 'evaluation_results.json')

# Load data and resources
@st.cache_data
def load_data():
    # Find the most recent labeled feedback file
    feedback_files = [f for f in os.listdir(DATA_DIR) if f.startswith('labeled_') and f.endswith('.csv')]
    if not feedback_files:
        st.error("No labeled feedback data found. Please run the pipeline first.")
        return None, None, None, None, None
    
    latest_file = max(feedback_files, key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)))
    feedback_path = os.path.join(DATA_DIR, latest_file)
    
    # Load feedback data
    df = load_feedback_data(feedback_path)
    
    # Convert date column to datetime if it exists
    if df is not None and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    
    # Load topic insights if available
    insights = None
    if os.path.exists(INSIGHTS_PATH):
        with open(INSIGHTS_PATH, 'r') as f:
            insights = json.load(f)
    
    # Load evaluation results if available
    eval_results = None
    if os.path.exists(EVAL_RESULTS_PATH):
        with open(EVAL_RESULTS_PATH, 'r') as f:
            eval_results = json.load(f)
    
    # Load topic summary if available
    topic_summary_files = [f for f in os.listdir(DATA_DIR) if f.startswith('topic_summary_') and f.endswith('.csv')]
    topic_summary = None
    if topic_summary_files:
        latest_summary = max(topic_summary_files, key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)))
        summary_path = os.path.join(DATA_DIR, latest_summary)
        if os.path.getsize(summary_path) > 0:
            try:
                topic_summary = pd.read_csv(summary_path)
            except pd.errors.EmptyDataError:
                st.warning(f"Warning: topic summary file {latest_summary} is empty. Skipping.")
            except Exception as e:
                st.error(f"Error loading topic summary file {latest_summary}: {e}")
        else:
            st.warning(f"Warning: topic summary file {latest_summary} is empty. Skipping.")
    return df, topic_summary, insights, eval_results, feedback_path

@st.cache_data
def load_evaluation_results():
    eval_path = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
    if not os.path.exists(eval_path):
        return None
    
    try:
        with open(eval_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading evaluation results: {e}")
        return None

# Load vector store for RAG
@st.cache_resource
def get_vector_store():
    if os.path.exists(VECTOR_STORE_DIR):
        try:
            return load_vector_store(VECTOR_STORE_DIR)
        except Exception as e:
            error_msg = str(e).lower()
            if "event loop" in error_msg or "asyncio" in error_msg or "already running" in error_msg:
                st.warning("Vector store loading failed due to async event loop issue. RAG functionality will be disabled.")
                return None
            else:
                st.error(f"Error loading vector store: {e}")
                return None
    return None

# Overview page
def show_overview(df, topic_summary=None, insights=None):
     st.header("📊 Overview")
     
     # Calculate summary statistics
     total_feedback = len(df)
     avg_nps = df['nps_score'].mean()
     promoters = len(df[df['nps_category'] == 'Promoter'])
     detractors = len(df[df['nps_category'] == 'Detractor'])
     passives = len(df[df['nps_category'] == 'Passive'])
     
     # Display summary metrics
     col1, col2, col3, col4 = st.columns(4)
     
     with col1:
         st.metric("Total Feedback", total_feedback)
     
     with col2:
         st.metric("Average NPS", f"{avg_nps:.1f}")
     
     with col3:
         st.metric("Promoters", f"{promoters} ({promoters/total_feedback*100:.1f}%)")
     
     with col4:
         st.metric("Detractors", f"{detractors} ({detractors/total_feedback*100:.1f}%)")
     
     # Empty-state cards for optional datasets
     info_cols = st.columns(2)
     with info_cols[0]:
         if topic_summary is None:
             st.info("Topic summary not found. Run the pipeline to generate topic summaries.")
     with info_cols[1]:
         if insights is None:
             st.info("Insights not found. Run the RAG insights pipeline to generate topic insights.")
     
     # NPS distribution
     st.subheader("NPS Distribution")
     
     # Create NPS distribution chart
     nps_counts = df['nps_score'].value_counts().sort_index()
     fig = px.bar(
         x=nps_counts.index,
         y=nps_counts.values,
         labels={'x': 'NPS Score', 'y': 'Count'},
         color=nps_counts.index,
         color_continuous_scale='RdYlGn',
         title='NPS Score Distribution'
     )
     st.plotly_chart(fig, use_container_width=True, key="nps_dist_chart")
     
     # Topic distribution
     st.subheader("Topic Distribution")
     
     # Create topic distribution chart
     topic_counts = df['topic'].value_counts().head(10)
     fig = px.bar(
         x=topic_counts.index,
         y=topic_counts.values,
         labels={'x': 'Topic', 'y': 'Count'},
         title='Top 10 Topics'
     )
     fig.update_layout(xaxis_tickangle=-45)
     st.plotly_chart(fig, use_container_width=True, key="topic_dist_chart")
     
     # Topic Summary Highlights (optional enrichment)
     if topic_summary is not None:
        st.subheader("Topic Summary Highlights")
        try:
            ts = topic_summary.copy()
            # Normalize column names if needed
            ts.columns = [c.strip().lower() for c in ts.columns]
            # Expecting columns like 'topic', 'count', 'avg_nps'
            cols_ok = all(col in ts.columns for col in ['topic', 'count'])
            if cols_ok:
                top_ts = ts.sort_values('count', ascending=False).head(5)
                st.dataframe(top_ts[['topic', 'count'] + (["avg_nps"] if 'avg_nps' in top_ts.columns else [])], use_container_width=True)
                # Optional: bar of avg_nps for top topics if available
                if 'avg_nps' in top_ts.columns:
                    fig_ts = px.bar(top_ts, x='topic', y='avg_nps', title='Top Topics by Avg NPS')
                    fig_ts.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_ts, use_container_width=True)
            else:
                with st.expander("View topic summary (raw)"):
                    st.dataframe(topic_summary, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render topic summary highlights: {e}")
    # AI Topic Insights (optional enrichment)
     if insights is not None:
        st.subheader("AI Topic Insights")
        try:
            # Try to surface a few insights aligned with top topics
            top_topics = df['topic'].value_counts().head(3).index.tolist()
            if isinstance(insights, dict):
                shown_any = False
                for t in top_topics:
                    val = insights.get(t)
                    if isinstance(val, dict):
                        # Prefer 'insights' or 'summary' fields if present
                        items = val.get('insights') or val.get('summary')
                        if isinstance(items, list) and items:
                            with st.expander(f"Insights for {t}"):
                                for it in items[:3]:
                                    st.write(f"- {it}")
                            shown_any = True
                        elif isinstance(items, str):
                            with st.expander(f"Insights for {t}"):
                                st.write(items)
                            shown_any = True
                    elif isinstance(val, list) and val:
                        with st.expander(f"Insights for {t}"):
                            for it in val[:3]:
                                st.write(f"- {it}")
                        shown_any = True
                # Fallback: show global_insights if present
                if not shown_any and 'global_insights' in insights and isinstance(insights['global_insights'], list):
                    with st.expander("Global Insights"):
                        for it in insights['global_insights'][:5]:
                            st.write(f"- {it}")
                if not shown_any and 'insights' in insights and isinstance(insights['insights'], list):
                    with st.expander("Insights"):
                        for it in insights['insights'][:5]:
                            st.write(f"- {it}")
            else:
                # As a last resort, show JSON
                with st.expander("Insights (raw)"):
                    st.json(insights)
        except Exception as e:
            st.warning(f"Could not render insights: {e}")
     # Feedback over time
     st.subheader("Feedback Over Time")
     
     # Group by date and count
     if 'date' in df.columns:
         time_df = df.groupby(df['date'].dt.date).size().reset_index(name='count')
         time_df.columns = ['date', 'count']
         
         # Create time series chart
         fig = px.line(
             time_df,
             x='date',
             y='count',
             labels={'date': 'Date', 'count': 'Feedback Count'},
             title='Feedback Volume Over Time'
         )
         st.plotly_chart(fig, use_container_width=True, key="feedback_time_chart")
     else:
         st.warning("Date information not available in the dataset.")

# Topic Insights page
def show_topic_insights(df, topic_summary=None, insights=None):
     st.header("🔍 Topic Insights")
     
     # Calculate topic statistics
     if topic_summary is not None:
        try:
            topic_stats = topic_summary.copy()
            topic_stats.columns = [c.strip().lower() for c in topic_stats.columns]
            # Ensure needed columns
            if 'topic' not in topic_stats.columns:
                raise ValueError('topic_summary missing topic column')
            if 'count' not in topic_stats.columns:
                # derive count from df if necessary
                df_counts = df.groupby('topic').size().reset_index(name='count')
                topic_stats = topic_stats.merge(df_counts, on='topic', how='left')
            if 'avg_nps' not in topic_stats.columns and 'nps_score' in df.columns:
                df_avg = df.groupby('topic')['nps_score'].mean().reset_index(name='avg_nps')
                topic_stats = topic_stats.merge(df_avg, on='topic', how='left')
            topic_stats = topic_stats.sort_values('count', ascending=False)
        except Exception:
            # fallback to computing from df
            topic_stats = df.groupby('topic').agg({'feedback_id': 'count', 'nps_score': 'mean'}).reset_index()
            topic_stats.columns = ['topic', 'count', 'avg_nps']
            topic_stats = topic_stats.sort_values('count', ascending=False)
     else:
         topic_stats = df.groupby('topic').agg({'feedback_id': 'count', 'nps_score': 'mean'}).reset_index()
         topic_stats.columns = ['topic', 'count', 'avg_nps']
         topic_stats = topic_stats.sort_values('count', ascending=False)
     
     # Topic selection
     selected_topic = st.selectbox(
         "Select a topic to explore",
         options=topic_stats['topic'].tolist(),
         index=0
     )
     
     # Display topic details
     st.subheader(f"Topic: {selected_topic}")
     
     # Get data for selected topic
     topic_df = df[df['topic'] == selected_topic]
     
     # Display metrics
     col1, col2, col3 = st.columns(3)
     
     with col1:
         st.metric("Feedback Count", len(topic_df))
     
     with col2:
         avg_nps = topic_df['nps_score'].mean()
         st.metric("Average NPS", f"{avg_nps:.1f}")
     
     with col3:
         # Calculate sentiment distribution
         promoters = len(topic_df[topic_df['nps_category'] == 'Promoter'])
         detractors = len(topic_df[topic_df['nps_category'] == 'Detractor'])
         passives = len(topic_df[topic_df['nps_category'] == 'Passive'])
         total = len(topic_df)
         
         # Display the most common sentiment
         if promoters >= detractors and promoters >= passives:
             sentiment = "Positive"
             value = f"{promoters/total*100:.1f}% Promoters"
         elif detractors >= promoters and detractors >= passives:
             sentiment = "Negative"
             value = f"{detractors/total*100:.1f}% Detractors"
         else:
             sentiment = "Neutral"
             value = f"{passives/total*100:.1f}% Passives"
         
         st.metric("Dominant Sentiment", sentiment, value)
     
     # Optional: show topic summary row if available
     if topic_summary is not None:
        try:
            ts = topic_summary.copy()
            ts.columns = [c.strip().lower() for c in ts.columns]
            row = ts[ts['topic'] == selected_topic]
            if not row.empty:
                with st.expander("Topic summary details"):
                    cols_to_show = [c for c in ['count', 'avg_nps'] if c in ts.columns]
                    st.write(row[['topic'] + cols_to_show].reset_index(drop=True))
        except Exception:
            pass
     # Display NPS distribution for this topic
     st.subheader("NPS Distribution")
     
     # Create NPS distribution chart for this topic
     nps_counts = topic_df['nps_score'].value_counts().sort_index()
     fig = px.bar(
         x=nps_counts.index,
         y=nps_counts.values,
         labels={'x': 'NPS Score', 'y': 'Count'},
         color=nps_counts.index,
         color_continuous_scale='RdYlGn',
         title=f'NPS Distribution for {selected_topic}'
     )
     st.plotly_chart(fig, use_container_width=True)
     
     # AI insights for this topic (optional enrichment)
     if insights is not None:
        try:
            if isinstance(insights, dict):
                val = insights.get(selected_topic)
                if isinstance(val, dict):
                    items = val.get('insights') or val.get('summary')
                    if isinstance(items, list) and items:
                        st.subheader("AI Insights for this Topic")
                        for it in items[:5]:
                            st.write(f"- {it}")
                    elif isinstance(items, str):
                        st.subheader("AI Insights for this Topic")
                        st.write(items)
                elif isinstance(val, list) and val:
                    st.subheader("AI Insights for this Topic")
                    for it in val[:5]:
                        st.write(f"- {it}")
            else:
                with st.expander("Insights (raw)"):
                    st.json(insights)
        except Exception as e:
            st.warning(f"Could not render topic insights: {e}")
     else:
         st.info("No AI insights available for topics.")
     # Display sample feedback
     st.subheader("Sample Feedback")
     
     # Create tabs for different sentiment categories
     tab1, tab2, tab3 = st.tabs(["Promoters", "Passives", "Detractors"])
     
     with tab1:
         promoter_df = topic_df[topic_df['nps_category'] == 'Promoter']
         if not promoter_df.empty:
             for _, row in promoter_df.head(5).iterrows():
                 st.markdown(f"**NPS {row['nps_score']}:** {row['feedback_text']}")
         else:
             st.write("No promoter feedback for this topic.")
     
     with tab2:
         passive_df = topic_df[topic_df['nps_category'] == 'Passive']
         if not passive_df.empty:
             for _, row in passive_df.head(5).iterrows():
                 st.markdown(f"**NPS {row['nps_score']}:** {row['feedback_text']}")
         else:
             st.write("No passive feedback for this topic.")
     
     with tab3:
         detractor_df = topic_df[topic_df['nps_category'] == 'Detractor']
         if not detractor_df.empty:
             for _, row in detractor_df.head(5).iterrows():
                 st.markdown(f"**NPS {row['nps_score']}:** {row['feedback_text']}")
         else:
             st.write("No detractor feedback for this topic.")

# Trend Analysis page
def show_trend_analysis(feedback_path):
    st.header("📈 Trend Analysis")
    
    try:
        # Detect trends with daily time window for small dataset
        trends_df = detect_trends(feedback_path, OUTPUT_DIR, time_window='day', min_count=2)
        
        # Display detected trends
        st.subheader("Detected Trends")
        
        if not trends_df.empty:
            # Display trends as a table
            st.dataframe(trends_df)
            
            # Show top trending topics
            st.subheader("Top Trending Topics")
            top_trends = trends_df.head(10)
            
            for idx, row in top_trends.iterrows():
                with st.expander(f"Topic: {row['topic']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if row['count_growth'] == float('inf'):
                            st.metric("Count Growth", "New Topic")
                        else:
                            st.metric("Count Growth", f"{row['count_growth']*100:.1f}%")
                    with col2:
                        if pd.isna(row['nps_change']) or row['nps_change'] is None:
                            st.metric("NPS Change", "N/A")
                        else:
                            st.metric("NPS Change", f"{row['nps_change']:.2f}")
                    with col3:
                        st.metric("Current Count", int(row['latest_count']))
        else:
            st.info("No significant trends detected in the current time period.")
        
        # Display visualizations if they exist
        st.subheader("Trend Visualizations")
        
        # Check for existing visualization files
        viz_files = {
            'top_growing_topics': os.path.join(OUTPUT_DIR, 'top_growing_topics.png'),
            'topic_volume_over_time': os.path.join(OUTPUT_DIR, 'topic_volume_over_time.png'),
            'nps_changes': os.path.join(OUTPUT_DIR, 'nps_changes.png')
        }
        
        # Create tabs for different visualizations
        tabs = st.tabs([
            "Top Growing Topics",
            "Topic Volume Over Time",
            "NPS Changes"
        ])
        
        with tabs[0]:
            if os.path.exists(viz_files['top_growing_topics']):
                st.image(viz_files['top_growing_topics'])
            else:
                st.write("Top growing topics visualization not available.")
        
        with tabs[1]:
            if os.path.exists(viz_files['topic_volume_over_time']):
                st.image(viz_files['topic_volume_over_time'])
            else:
                st.write("Topic volume over time visualization not available.")
        
        with tabs[2]:
            if os.path.exists(viz_files['nps_changes']):
                st.image(viz_files['nps_changes'])
            else:
                st.write("NPS changes visualization not available.")
        
    except Exception as e:
        st.error(f"Error in trend analysis: {e}")

# Cohort Analysis page
def show_cohort_analysis(feedback_path):
    st.header("👥 Cohort Analysis")
    
    try:
        # Perform cohort analysis
        cohort_results = perform_cohort_analysis(feedback_path, output_dir=OUTPUT_DIR)
        
        if cohort_results is not None and len(cohort_results['cohort_sizes']) > 0:
            # Display cohort summary
            st.subheader("Cohort Summary")
            
            # Create cohort summary table from the results
            cohort_data = []
            for cohort in cohort_results['cohort_sizes'].keys():
                cohort_data.append({
                    'Cohort': cohort,
                    'Size': cohort_results['cohort_sizes'][cohort],
                    'Avg NPS': cohort_results['cohort_nps'].get(cohort, 0)
                })
            
            cohort_df = pd.DataFrame(cohort_data)
            st.dataframe(cohort_df, use_container_width=True)
            
            # Display top topics by cohort
            st.subheader("Top Topics by Cohort")
            for cohort, topics in cohort_results['cohort_topics'].items():
                with st.expander(f"Cohort: {cohort}"):
                    if topics:
                        topics_df = pd.DataFrame(topics)
                        st.dataframe(topics_df)
                    else:
                        st.info("No topics found for this cohort.")
            
            # Check for visualization files in output directory
            viz_files = {
                'cohort_sizes': os.path.join(OUTPUT_DIR, 'cohort_sizes.png'),
                'cohort_nps': os.path.join(OUTPUT_DIR, 'cohort_nps.png'),
                'topic_distribution': os.path.join(OUTPUT_DIR, 'topic_distribution_by_cohort.png'),
                'nps_by_topic': os.path.join(OUTPUT_DIR, 'nps_by_topic_and_cohort.png')
            }
            
            # Create tabs for different visualizations
            tabs = st.tabs([
                "Cohort Sizes",
                "Cohort NPS",
                "Topic Distribution",
                "NPS by Topic and Cohort"
            ])
            
            with tabs[0]:
                if os.path.exists(viz_files['cohort_sizes']):
                    st.image(viz_files['cohort_sizes'])
                else:
                    st.write("Cohort sizes visualization not available.")
            
            with tabs[1]:
                if os.path.exists(viz_files['cohort_nps']):
                    st.image(viz_files['cohort_nps'])
                else:
                    st.write("Cohort NPS visualization not available.")
            
            with tabs[2]:
                if os.path.exists(viz_files['topic_distribution']):
                    st.image(viz_files['topic_distribution'])
                else:
                    st.write("Topic distribution visualization not available.")
            
            with tabs[3]:
                if os.path.exists(viz_files['nps_by_topic']):
                    st.image(viz_files['nps_by_topic'])
                else:
                    st.write("NPS by topic visualization not available.")
            
            # Display cohort metrics
            st.subheader("Cohort Metrics")
            
            # Create metrics for each cohort
            cohorts = list(cohort_results['cohort_sizes'].keys())
            if len(cohorts) <= 4:
                cols = st.columns(len(cohorts))
                for i, cohort in enumerate(cohorts):
                    with cols[i]:
                        st.metric(
                            f"Cohort {cohort}",
                            f"Size: {cohort_results['cohort_sizes'][cohort]}",
                            f"NPS: {cohort_results['cohort_nps'].get(cohort, 0):.2f}"
                        )
            else:
                # If too many cohorts, display in a different format
                for cohort in cohorts:
                    st.write(f"**{cohort}:** Size: {cohort_results['cohort_sizes'][cohort]}, NPS: {cohort_results['cohort_nps'].get(cohort, 0):.2f}")
        else:
            st.warning("Not enough data for cohort analysis with the current settings.")
    
    except Exception as e:
        st.error(f"Error in cohort analysis: {e}")

# Feedback Explorer page
def show_feedback_explorer(df):
    st.header("🔍 Feedback Explorer")
    
    # Filters
    st.subheader("Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # NPS score filter
        nps_range = st.slider(
            "NPS Score Range",
            min_value=int(df['nps_score'].min()),
            max_value=int(df['nps_score'].max()),
            value=(int(df['nps_score'].min()), int(df['nps_score'].max()))
        )
    
    with col2:
        # Topic filter
        selected_topics = st.multiselect(
            "Topics",
            options=df['topic'].unique().tolist(),
            default=df['topic'].unique().tolist()
        )
    
    with col3:
        # Date filter (if available)
        if 'date' in df.columns:
            date_range = st.date_input(
                "Date Range",
                value=(df['date'].min(), df['date'].max()),
                min_value=df['date'].min(),
                max_value=df['date'].max()
            )
        else:
            date_range = None
    
    # Apply filters
    filtered_df = df[
        (df['nps_score'] >= nps_range[0]) &
        (df['nps_score'] <= nps_range[1]) &
        (df['topic'].isin(selected_topics))
    ]
    
    if date_range and 'date' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) &
            (filtered_df['date'].dt.date <= date_range[1])
        ]
    
    # Display filtered results
    st.subheader(f"Filtered Results ({len(filtered_df)} items)")
    
    # Search box
    search_term = st.text_input("Search in feedback text", "")
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df['feedback_text'].str.contains(search_term, case=False, na=False)
        ]
    
    # Display results
    for _, row in filtered_df.head(50).iterrows():
        with st.expander(f"NPS {row['nps_score']} - {row['topic']}"):
            st.write(f"**Feedback:** {row['feedback_text']}")
            if 'date' in row:
                st.write(f"**Date:** {row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'}")
            st.write(f"**Category:** {row['nps_category']}")

# RAG Q&A page
def show_rag_qa(df):
    st.header("💬 Ask Questions About Your Feedback")
    
    # Load vector store
    vector_store = get_vector_store()
    if vector_store is None:
        st.error("Vector store not available. This could be due to:")
        st.write("• Vector store not found - please run the RAG pipeline first")
        st.write("• Event loop issue with Google Generative AI embeddings in Streamlit")
        st.info("Try using OpenAI embeddings instead by setting API_PROVIDER=openai in your .env file")
        return
    
    # User query input
    query = st.text_input("Ask a question about your customer feedback", "")
    
    # Example questions
    st.markdown("### Example questions")
    example_questions = [
        "What are the top issues customers are complaining about?",
        "Why are customers giving low NPS scores?",
        "What features do customers like the most?",
        "What improvements would have the biggest impact on customer satisfaction?",
        "How can we improve the mobile app experience?"
    ]
    
    # Create columns for example questions
    cols = st.columns(len(example_questions))
    clicked_question = None
    
    # Add buttons for example questions
    for i, (col, question) in enumerate(zip(cols, example_questions)):
        if col.button(f"Q{i+1}", key=f"q{i}"):
            clicked_question = question
    
    # Display the example questions below the buttons
    for i, question in enumerate(example_questions):
        st.markdown(f"**Q{i+1}:** {question}")
    
    # If an example question is clicked, use it as the query
    if clicked_question:
        query = clicked_question
        st.text_input("Ask a question about your customer feedback", query, key="updated_query")
    
    # Generate answer if query is provided
    if query:
        with st.spinner("Generating answer..."):
            result = generate_rag_answer(query, vector_store)
        
        if result:
            st.markdown("### Answer")
            st.markdown(result['answer'])
            
            st.markdown("### Sources")
            for i, source in enumerate(result['sources'], 1):
                with st.expander(f"Source {i}: {source['topic']} (NPS: {source['nps_score']})"): 
                    st.write(source['feedback_text'])

# Evaluation Results page
def show_evaluation_results(eval_results):
    st.header("📋 Evaluation Results")
    
    if eval_results is None:
        st.warning("No evaluation results available. Please run the evaluation pipeline first.")
        st.info("To generate evaluation results, run: `python src/evaluation/evaluate_rag.py`")
        return
    
    # Check if query_results exists
    if 'query_results' not in eval_results:
        st.error("Invalid evaluation results format. Missing 'query_results' key.")
        return
    
    # Display metrics
    st.subheader("Evaluation Metrics")
    metrics = eval_results.get('metrics', eval_results.get('evaluation_metrics', {}))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Groundedness", f"{metrics.get('groundedness', 0):.2f}/1.0")
    
    with col2:
        st.metric("Relevance", f"{metrics.get('relevance', 0):.2f}/1.0")
    
    with col3:
        st.metric("Overall Score", f"{metrics.get('overall', 0):.2f}/1.0")
    
    # Display total queries if results are available
    if 'results' in eval_results:
        st.metric("Total Queries", len(eval_results['results']))
    
    # Create a radar chart of metrics
    st.subheader("Metrics Visualization")
    
    # Prepare data for radar chart
    categories = ['Groundedness', 'Relevance', 'Overall']
    values = [metrics.get('groundedness', 0), metrics.get('relevance', 0), metrics.get('overall', 0)]
    
    # Create radar chart
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    
    # Compute angle for each category
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values.append(values[0])  # Close the loop
    angles.append(angles[0])  # Close the loop
    categories.append(categories[0])  # Close the loop
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, label='Scores')
    ax.fill(angles, values, alpha=0.25)
    
    # Set category labels
    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Evaluation Metrics', size=15, y=1.1)
    
    st.pyplot(fig)
    
    # Display evaluation results
    st.subheader("Evaluation Examples")
    
    for i, result in enumerate(eval_results['query_results'], 1):
        with st.expander(f"Example {i}: {result['query']}"):
            st.markdown("**Query:**")
            st.write(result['query'])
            
            st.markdown("**Answer:**")
            st.write(result['answer'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Groundedness", f"{result['groundedness_score']:.2f}")
            with col2:
                st.metric("Relevance", f"{result['relevance_score']:.2f}")
            with col3:
                st.metric("Correctness", f"{result['correctness_score']:.2f}")

# Main function
def main():
    st.title("Customer Feedback Analysis Dashboard")
    
    # Load data
    df, topic_summary, insights, eval_results, feedback_path = load_data()
    if df is None:
        st.error("No data found. Please run the pipeline first.")
        return
    
    # Load evaluation results if not already loaded
    if eval_results is None:
        eval_results = load_evaluation_results()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Overview", "Topic Insights", "Feedback Explorer", "RAG Q&A", "Evaluation Results", "Trend Analysis", "Cohort Analysis"]
    )
    
    # Display selected page
    if page == "Overview":
        show_overview(df)
    elif page == "Topic Insights":
        show_topic_insights(df)
    elif page == "Feedback Explorer":
        show_feedback_explorer(df)
    elif page == "RAG Q&A":
        show_rag_qa(df)
    elif page == "Evaluation Results":
        show_evaluation_results(eval_results)
    elif page == "Trend Analysis":
        show_trend_analysis(feedback_path)
    elif page == "Cohort Analysis":
        show_cohort_analysis(feedback_path)

# Run the app
if __name__ == "__main__":
    main()