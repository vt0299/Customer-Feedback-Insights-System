import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_feedback_data

logger = logging.getLogger(__name__)

def perform_cohort_analysis(feedback_file, cohort_column=None, output_dir=None):
    """
    Perform cohort analysis on feedback data
    
    Args:
        feedback_file: Path to labeled feedback CSV file
        cohort_column: Column to use for cohort segmentation (if None, will use date-based cohorts)
        output_dir: Directory to save cohort analysis results
        
    Returns:
        Dictionary with cohort analysis results
    """
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.dirname(feedback_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load feedback data
    df = load_feedback_data(feedback_file)
    
    if df is None or len(df) == 0:
        logger.error("No data loaded from feedback file")
        return None
    
    # Ensure date column is datetime
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        if len(df) == 0:
            logger.error("No valid dates found in the data")
            return None
    except Exception as e:
        logger.error(f"Error converting date column: {e}")
        return None
    
    # If no cohort column is specified, create date-based cohorts (by month)
    if cohort_column is None:
        df['cohort'] = df['date'].dt.to_period('M').dt.to_timestamp().dt.strftime('%Y-%m')
        cohort_column = 'cohort'
    elif cohort_column not in df.columns:
        logger.error(f"Cohort column '{cohort_column}' not found in feedback data")
        return None
    
    # Create cohort groups
    cohorts = df[cohort_column].unique()
    
    # Initialize results
    results = {
        'cohort_nps': {},
        'cohort_topics': {},
        'cohort_sizes': {}
    }
    
    # Calculate cohort sizes
    cohort_sizes = df.groupby(cohort_column).size()
    results['cohort_sizes'] = cohort_sizes.to_dict()
    
    # Calculate NPS by cohort
    cohort_nps = df.groupby(cohort_column)['nps_score'].mean()
    results['cohort_nps'] = cohort_nps.to_dict()
    
    # Calculate top topics by cohort
    for cohort in cohorts:
        cohort_df = df[df[cohort_column] == cohort]
        topic_counts = cohort_df.groupby('topic').size().reset_index(name='count')
        topic_counts = topic_counts.sort_values('count', ascending=False)
        
        # Calculate topic NPS
        topic_nps = cohort_df.groupby('topic')['nps_score'].mean().reset_index(name='avg_nps')
        
        # Merge counts and NPS
        topic_stats = pd.merge(topic_counts, topic_nps, on='topic')
        
        # Store top 5 topics for this cohort
        results['cohort_topics'][cohort] = topic_stats.head(5).to_dict('records')
    
    # Generate visualizations
    generate_cohort_visualizations(df, cohort_column, results, output_dir)
    
    # Save results to CSV
    cohort_file = os.path.join(output_dir, 'cohort_analysis.csv')
    
    # Create a DataFrame for cohort summary
    cohort_summary = pd.DataFrame({
        'cohort': list(results['cohort_sizes'].keys()),
        'size': list(results['cohort_sizes'].values()),
        'avg_nps': list(results['cohort_nps'].values())
    })
    
    cohort_summary.to_csv(cohort_file, index=False)
    logger.info(f"Cohort analysis saved to {cohort_file}")
    
    return results

def generate_cohort_visualizations(df, cohort_column, results, output_dir):
    """
    Generate visualizations for cohort analysis
    """
    # 1. Cohort sizes
    plt.figure(figsize=(12, 6))
    cohort_sizes = pd.Series(results['cohort_sizes'])
    cohort_sizes.plot(kind='bar')
    plt.xlabel('Cohort')
    plt.ylabel('Count')
    plt.title('Cohort Sizes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cohort_sizes.png'))
    plt.close()
    
    # 2. Cohort NPS
    plt.figure(figsize=(12, 6))
    cohort_nps = pd.Series(results['cohort_nps'])
    cohort_nps.plot(kind='bar')
    plt.xlabel('Cohort')
    plt.ylabel('Average NPS')
    plt.title('Average NPS by Cohort')
    plt.xticks(rotation=45)
    plt.axhline(y=cohort_nps.mean(), color='red', linestyle='--', label='Overall Average')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cohort_nps.png'))
    plt.close()
    
    # 3. Topic distribution by cohort
    # Get top 5 topics overall
    top_topics = df.groupby('topic').size().nlargest(5).index.tolist()
    
    # Create a pivot table of cohort vs topic
    topic_pivot = pd.pivot_table(
        df[df['topic'].isin(top_topics)],
        values='feedback_id',
        index=cohort_column,
        columns='topic',
        aggfunc='count',
        fill_value=0
    )
    
    # Normalize by cohort size - convert to float first to avoid dtype issues
    topic_pivot = topic_pivot.astype(float)
    for cohort in topic_pivot.index:
        topic_pivot.loc[cohort] = topic_pivot.loc[cohort] / results['cohort_sizes'][cohort]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(topic_pivot, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title('Topic Distribution by Cohort (Normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_distribution_by_cohort.png'))
    plt.close()
    
    # 4. NPS by topic and cohort
    # Create a pivot table of cohort vs topic for NPS
    nps_pivot = pd.pivot_table(
        df[df['topic'].isin(top_topics)],
        values='nps_score',
        index=cohort_column,
        columns='topic',
        aggfunc='mean',
        fill_value=np.nan
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(nps_pivot, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=10)
    plt.title('Average NPS by Topic and Cohort')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nps_by_topic_and_cohort.png'))
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform cohort analysis on feedback data')
    parser.add_argument('--input', required=True, help='Path to labeled feedback CSV file')
    parser.add_argument('--cohort-column', help='Column to use for cohort segmentation')
    parser.add_argument('--output-dir', help='Directory to save cohort analysis results')
    
    args = parser.parse_args()
    
    # Perform cohort analysis
    results = perform_cohort_analysis(
        args.input,
        args.cohort_column,
        args.output_dir
    )
    
    # Print summary
    if results is None:
        print("Cohort analysis failed")
    else:
        print("\nCohort Analysis Summary:")
        for cohort, size in results['cohort_sizes'].items():
            nps = results['cohort_nps'].get(cohort, 'N/A')
            if isinstance(nps, (int, float)):
                nps = f"{nps:.2f}"
            print(f"  {cohort}: {size} feedbacks, Avg NPS: {nps}")
            
            print("    Top topics:")
            for topic in results['cohort_topics'].get(cohort, [])[:3]:
                print(f"      {topic['topic']}: {topic['count']} mentions, Avg NPS: {topic['avg_nps']:.2f}")

if __name__ == "__main__":
    main()