import argparse
import pandas as pd
import sys
import os
import logging

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_feedback_data, save_feedback_data

logger = logging.getLogger(__name__)

def clean_feedback(df):
    """
    Clean and preprocess feedback data
    """
    # Remove rows with missing feedback text
    df = df.dropna(subset=['feedback_text'])
    
    # Handle different column naming conventions
    # Map rating to nps_score if needed
    if 'rating' in df.columns and 'nps_score' not in df.columns:
        df['nps_score'] = df['rating']
    
    # Map timestamp to date if needed
    if 'timestamp' in df.columns and 'date' not in df.columns:
        df['date'] = df['timestamp']
    
    # Convert NPS scores to integers if they're not already
    if df['nps_score'].dtype != 'int64':
        df['nps_score'] = df['nps_score'].astype(int)
    
    # Ensure NPS scores are in the valid range (0-10)
    df = df[(df['nps_score'] >= 0) & (df['nps_score'] <= 10)]
    
    # Convert date to datetime if it's not already
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])
    
    # Add NPS category if not already present
    if 'nps_category' not in df.columns:
        df['nps_category'] = df['nps_score'].apply(
            lambda x: 'detractor' if x <= 6 else ('passive' if x <= 8 else 'promoter')
        )
    
    logger.info(f"Cleaned feedback data: {len(df)} valid entries")
    return df

def ingest_feedback(input_file, output_file=None):
    """
    Ingest feedback data from input file, clean it, and save to output file
    """
    # Load data
    df = load_feedback_data(input_file)
    if df is None:
        return None
    
    # Clean data
    df = clean_feedback(df)
    
    # Save cleaned data if output file is specified
    if output_file:
        save_feedback_data(df, output_file)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Ingest and clean customer feedback data')
    parser.add_argument('--input', required=True, help='Input CSV file with feedback data')
    parser.add_argument('--output', help='Output CSV file for cleaned data')
    args = parser.parse_args()
    
    # If output is not specified, use a default name based on the input file
    if not args.output:
        input_dir = os.path.dirname(args.input)
        input_filename = os.path.basename(args.input)
        output_filename = f"cleaned_{input_filename}"
        args.output = os.path.join(input_dir, output_filename)
    
    # Ingest and clean feedback
    df = ingest_feedback(args.input, args.output)
    
    if df is not None:
        # Print summary statistics
        print(f"\nFeedback Summary:")
        print(f"Total entries: {len(df)}")
        print(f"NPS distribution:")
        print(f"  Detractors (0-6): {len(df[df['nps_category'] == 'detractor'])}")
        print(f"  Passives (7-8): {len(df[df['nps_category'] == 'passive'])}")
        print(f"  Promoters (9-10): {len(df[df['nps_category'] == 'promoter'])}")
        print(f"Average NPS score: {df['nps_score'].mean():.2f}")
        print(f"\nCleaned data saved to: {args.output}")

if __name__ == "__main__":
    main()