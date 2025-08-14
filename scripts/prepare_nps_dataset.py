#!/usr/bin/env python3
"""
Script to prepare NPS dataset for the feedback analysis project.
Combines customer and score data, adds synthetic feedback text and categories.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_feedback_text(score, is_premier, is_spam):
    """Generate synthetic feedback text based on NPS score and customer attributes."""
    
    if is_spam:
        return "This is spam feedback with no real content."
    
    # Feedback templates based on NPS categories
    promoter_feedback = [
        "Excellent service! I would definitely recommend this to others.",
        "Outstanding product quality and customer support.",
        "Love the features and ease of use. Highly recommended!",
        "Best in class solution. Very satisfied with the experience.",
        "Amazing product! Exceeded my expectations completely.",
        "Fantastic service and great value for money.",
        "Perfect solution for our needs. Will definitely continue using.",
        "Exceptional quality and reliability. Highly recommend!"
    ]
    
    passive_feedback = [
        "Good product overall, but could use some improvements.",
        "Decent service, meets basic requirements.",
        "It's okay, does what it's supposed to do.",
        "Average experience, nothing particularly outstanding.",
        "Satisfactory product with room for enhancement.",
        "Works fine but competitors might offer better features.",
        "Reasonable quality but pricing could be better.",
        "Good enough for now, might consider alternatives later."
    ]
    
    detractor_feedback = [
        "Very disappointed with the service quality.",
        "Poor customer support and frequent issues.",
        "Would not recommend due to reliability problems.",
        "Overpriced for the value provided.",
        "Terrible experience, looking for alternatives.",
        "Many bugs and technical issues encountered.",
        "Customer service is unresponsive and unhelpful.",
        "Product doesn't meet expectations at all.",
        "Frequent downtime and poor performance.",
        "Difficult to use and lacks important features."
    ]
    
    # Select feedback based on NPS category
    if score >= 9:  # Promoters
        feedback = random.choice(promoter_feedback)
        if is_premier:
            feedback += " The premium features are especially valuable."
    elif score >= 7:  # Passives
        feedback = random.choice(passive_feedback)
        if is_premier:
            feedback += " Expected more from the premium plan."
    else:  # Detractors
        feedback = random.choice(detractor_feedback)
        if is_premier:
            feedback += " Considering canceling premium subscription."
    
    return feedback

def generate_category(score, is_premier):
    """Generate category based on score and customer type."""
    categories = {
        'promoter': ['Product Quality', 'Customer Service', 'Features', 'Value'],
        'passive': ['Pricing', 'Features', 'Support', 'Usability'],
        'detractor': ['Technical Issues', 'Support', 'Pricing', 'Performance']
    }
    
    if score >= 9:
        category_type = 'promoter'
    elif score >= 7:
        category_type = 'passive'
    else:
        category_type = 'detractor'
    
    base_category = random.choice(categories[category_type])
    
    if is_premier:
        return f"Premium - {base_category}"
    else:
        return base_category

def main():
    """Main function to process and combine the datasets."""
    
    # Load the datasets
    print("Loading customer and score datasets...")
    customers_df = pd.read_csv('data/customer.csv')
    scores_df = pd.read_csv('data/score.csv')
    
    print(f"Loaded {len(customers_df)} customers and {len(scores_df)} scores")
    
    # Merge the datasets
    print("Merging datasets...")
    merged_df = scores_df.merge(customers_df, left_on='customer_id', right_on='id', suffixes=('_score', '_customer'))
    
    # Rename columns to match expected format
    merged_df = merged_df.rename(columns={
        'id_score': 'feedback_id',
        'customer_id': 'customer_id',
        'created_at_score': 'timestamp',
        'score': 'rating',
        'created_at_customer': 'customer_signup_date'
    })
    
    # Generate synthetic feedback text and categories
    print("Generating synthetic feedback text and categories...")
    merged_df['feedback_text'] = merged_df.apply(
        lambda row: generate_feedback_text(row['rating'], row['is_premier'], row['is_spam']), 
        axis=1
    )
    
    merged_df['category'] = merged_df.apply(
        lambda row: generate_category(row['rating'], row['is_premier']), 
        axis=1
    )
    
    # Add NPS classification
    def classify_nps(rating):
        if rating >= 9:
            return 'Promoter'
        elif rating >= 7:
            return 'Passive'
        else:
            return 'Detractor'
    
    merged_df['nps_category'] = merged_df['rating'].apply(classify_nps)
    
    # Convert timestamps to proper datetime format
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    merged_df['customer_signup_date'] = pd.to_datetime(merged_df['customer_signup_date'])
    
    # Select and reorder columns for final dataset
    final_columns = [
        'feedback_id', 'customer_id', 'timestamp', 'rating', 
        'feedback_text', 'category', 'nps_category', 
        'is_premier', 'is_spam', 'customer_signup_date'
    ]
    
    final_df = merged_df[final_columns]
    
    # Save the processed dataset
    output_file = 'data/nps_feedback_dataset.csv'
    print(f"Saving processed dataset to {output_file}...")
    final_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total feedback records: {len(final_df)}")
    print(f"Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
    print(f"Unique customers: {final_df['customer_id'].nunique()}")
    print("\nNPS Distribution:")
    print(final_df['nps_category'].value_counts())
    print("\nRating Distribution:")
    print(final_df['rating'].value_counts().sort_index())
    print("\nCategory Distribution:")
    print(final_df['category'].value_counts())
    print("\nPremium vs Regular customers:")
    print(final_df['is_premier'].value_counts())
    
    # Calculate NPS score
    promoters = len(final_df[final_df['nps_category'] == 'Promoter'])
    detractors = len(final_df[final_df['nps_category'] == 'Detractor'])
    total = len(final_df)
    nps_score = ((promoters - detractors) / total) * 100
    print(f"\nOverall NPS Score: {nps_score:.1f}")
    
    print(f"\nDataset successfully prepared and saved to {output_file}")
    print("You can now use this dataset for data ingestion in your feedback analysis project.")

if __name__ == "__main__":
    main()