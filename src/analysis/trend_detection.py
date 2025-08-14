import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
from datetime import datetime, timedelta
from collections import Counter

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_feedback_data

logger = logging.getLogger(__name__)

def detect_trends(feedback_file, output_dir=None, time_window='month', min_count=5):
    """
    Detect trends in feedback data over time
    
    Args:
        feedback_file: Path to labeled feedback CSV file
        output_dir: Directory to save trend analysis results
        time_window: Time window for aggregation ('day', 'week', 'month', 'quarter')
        min_count: Minimum count to consider a topic trending
        
    Returns:
        DataFrame with trend analysis results
    """
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.dirname(feedback_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load feedback data
    df = load_feedback_data(feedback_file)
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create time window column
    if time_window == 'day':
        df['time_window'] = df['date'].dt.date
    elif time_window == 'week':
        df['time_window'] = df['date'].dt.to_period('W').dt.start_time.dt.date
    elif time_window == 'month':
        df['time_window'] = df['date'].dt.to_period('M').dt.start_time.dt.date
    elif time_window == 'quarter':
        df['time_window'] = df['date'].dt.to_period('Q').dt.start_time.dt.date
    else:
        raise ValueError(f"Invalid time_window: {time_window}. Must be one of: day, week, month, quarter")
    
    # Group by time window and topic
    topic_counts = df.groupby(['time_window', 'topic']).size().reset_index(name='count')
    
    # Calculate average NPS by time window and topic
    topic_nps = df.groupby(['time_window', 'topic'])['nps_score'].mean().reset_index(name='avg_nps')
    
    # Merge counts and NPS
    topic_trends = pd.merge(topic_counts, topic_nps, on=['time_window', 'topic'])
    
    # Calculate trend metrics
    time_windows = sorted(df['time_window'].unique())
    
    # Initialize trend results
    trend_results = []
    
    # We need at least 2 time windows to detect trends
    if len(time_windows) < 2:
        logger.warning("Not enough time windows to detect trends")
        return pd.DataFrame()
    
    # Get the most recent time window
    latest_window = time_windows[-1]
    
    # Get the previous time window
    previous_window = time_windows[-2]
    
    # Get topics in the latest window
    latest_topics = topic_trends[topic_trends['time_window'] == latest_window]
    
    # Get topics in the previous window
    previous_topics = topic_trends[topic_trends['time_window'] == previous_window]
    
    # Calculate growth for each topic
    for _, row in latest_topics.iterrows():
        topic = row['topic']
        latest_count = row['count']
        latest_nps = row['avg_nps']
        
        # Skip topics with count below threshold
        if latest_count < min_count:
            continue
        
        # Find the same topic in the previous window
        prev_row = previous_topics[previous_topics['topic'] == topic]
        
        if len(prev_row) > 0:
            # Topic existed in previous window
            prev_count = prev_row.iloc[0]['count']
            prev_nps = prev_row.iloc[0]['avg_nps']
            
            # Calculate growth
            count_growth = (latest_count - prev_count) / prev_count if prev_count > 0 else float('inf')
            nps_change = latest_nps - prev_nps
            
            trend_results.append({
                'topic': topic,
                'latest_count': latest_count,
                'previous_count': prev_count,
                'count_growth': count_growth,
                'latest_nps': latest_nps,
                'previous_nps': prev_nps,
                'nps_change': nps_change,
                'is_new': False
            })
        else:
            # New topic
            trend_results.append({
                'topic': topic,
                'latest_count': latest_count,
                'previous_count': 0,
                'count_growth': float('inf'),
                'latest_nps': latest_nps,
                'previous_nps': None,
                'nps_change': None,
                'is_new': True
            })
    
    # Convert to DataFrame
    trend_df = pd.DataFrame(trend_results)
    
    # Sort by count growth (descending)
    if not trend_df.empty:
        trend_df = trend_df.sort_values('count_growth', ascending=False)
    
    # Save trend results
    trend_file = os.path.join(output_dir, 'trend_analysis.csv')
    trend_df.to_csv(trend_file, index=False)
    logger.info(f"Trend analysis saved to {trend_file}")
    
    # Generate visualizations
    if not trend_df.empty:
        generate_trend_visualizations(trend_df, topic_trends, output_dir)
    
    return trend_df

def generate_trend_visualizations(trend_df, topic_trends, output_dir):
    """
    Generate visualizations for trend analysis
    """
    # 1. Top growing topics
    plt.figure(figsize=(12, 6))
    
    # Filter out infinite growth (new topics)
    growing_topics = trend_df[~trend_df['is_new']].head(10)
    
    if not growing_topics.empty:
        plt.barh(growing_topics['topic'], growing_topics['count_growth'])
        plt.xlabel('Growth Rate')
        plt.ylabel('Topic')
        plt.title('Top Growing Topics')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_growing_topics.png'))
        plt.close()
    
    # 2. New topics
    new_topics = trend_df[trend_df['is_new']]
    
    if not new_topics.empty:
        plt.figure(figsize=(12, 6))
        plt.barh(new_topics['topic'], new_topics['latest_count'])
        plt.xlabel('Count')
        plt.ylabel('Topic')
        plt.title('New Topics')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'new_topics.png'))
        plt.close()
    
    # 3. Topic volume over time
    # Get top 5 topics by total volume
    top_topics = topic_trends.groupby('topic')['count'].sum().nlargest(5).index.tolist()
    
    # Filter for top topics
    top_topic_trends = topic_trends[topic_trends['topic'].isin(top_topics)]
    
    # Pivot for plotting
    pivot_df = top_topic_trends.pivot(index='time_window', columns='topic', values='count')
    
    plt.figure(figsize=(12, 6))
    pivot_df.plot(marker='o', ax=plt.gca())
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('Topic Volume Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_volume_over_time.png'))
    plt.close()
    
    # 4. NPS changes
    plt.figure(figsize=(12, 6))
    
    # Filter out new topics and sort by NPS change
    nps_change_topics = trend_df[~trend_df['is_new']].sort_values('nps_change').head(10)
    
    if not nps_change_topics.empty:
        plt.barh(nps_change_topics['topic'], nps_change_topics['nps_change'])
        plt.xlabel('NPS Change')
        plt.ylabel('Topic')
        plt.title('Topics with Largest NPS Changes')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'nps_changes.png'))
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect trends in feedback data')
    parser.add_argument('--input', required=True, help='Path to labeled feedback CSV file')
    parser.add_argument('--output-dir', help='Directory to save trend analysis results')
    parser.add_argument('--time-window', choices=['day', 'week', 'month', 'quarter'], default='month',
                        help='Time window for aggregation')
    parser.add_argument('--min-count', type=int, default=5, help='Minimum count to consider a topic trending')
    
    args = parser.parse_args()
    
    # Detect trends
    trend_df = detect_trends(
        args.input,
        args.output_dir,
        args.time_window,
        args.min_count
    )
    
    # Print summary
    if trend_df.empty:
        print("No trends detected")
    else:
        print("\nTop Growing Topics:")
        growing_topics = trend_df[~trend_df['is_new']].head(5)
        for _, row in growing_topics.iterrows():
            growth = row['count_growth']
            growth_str = f"{growth:.2f}x" if growth != float('inf') else "New"
            print(f"  {row['topic']}: {growth_str} growth, {row['latest_count']} mentions, NPS change: {row['nps_change']:.2f}")
        
        print("\nNew Topics:")
        new_topics = trend_df[trend_df['is_new']].head(5)
        for _, row in new_topics.iterrows():
            print(f"  {row['topic']}: {row['latest_count']} mentions, NPS: {row['latest_nps']:.2f}")

if __name__ == "__main__":
    main()