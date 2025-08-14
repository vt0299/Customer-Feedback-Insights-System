# NPS Feedback Dataset Guide

## Overview

This guide explains how to use the comprehensive NPS (Net Promoter Score) dataset that has been prepared for your feedback analysis project. The dataset contains over 1.5 million customer feedback records with ratings, synthetic feedback text, and rich metadata.

## Dataset Information

### Source
- **Original Data**: [sixhobbits/nps-sample-data](https://github.com/sixhobbits/nps-sample-data) <mcreference link="https://github.com/sixhobbits/nps-sample-data" index="1">1</mcreference>
- **Processed Dataset**: `data/nps_feedback_dataset.csv`
- **Total Records**: 1,577,578 feedback entries
- **Date Range**: January 1, 2018 - December 30, 2018
- **Unique Customers**: 188,323
- **Overall NPS Score**: 59.0 (Excellent)

### Dataset Structure

The processed dataset contains the following columns:

| Column | Description | Example |
|--------|-------------|----------|
| `feedback_id` | Unique identifier for each feedback entry | 1, 2, 3... |
| `customer_id` | Unique customer identifier | 1, 2, 3... |
| `timestamp` | When the feedback was submitted | 2018-01-01 05:27:28 |
| `rating` | NPS score (0-10 scale) | 0, 5, 10 |
| `feedback_text` | Synthetic feedback text based on rating | "Excellent service! I would definitely recommend..." |
| `category` | Feedback category | Product Quality, Customer Service, Technical Issues |
| `nps_category` | NPS classification | Promoter, Passive, Detractor |
| `is_premier` | Whether customer has premium plan | True, False |
| `is_spam` | Whether feedback is identified as spam | True, False |
| `customer_signup_date` | When customer signed up | 2018-01-01 15:06:53 |

### NPS Distribution

- **Promoters (9-10)**: 1,192,264 (75.6%)
- **Passives (7-8)**: 123,171 (7.8%)
- **Detractors (0-6)**: 262,143 (16.6%)

### Category Distribution

Top feedback categories:
- Features: 292,719
- Product Quality: 263,801
- Customer Service: 263,102
- Value: 262,850
- Pricing: 91,797
- Support: 91,271
- Performance: 62,175
- Technical Issues: 61,447

## How to Use This Dataset

### 1. Data Ingestion

To use this dataset with your existing pipeline, simply update your data ingestion script to point to the new file:

```python
# In your ingestion script
feedback_file = 'data/nps_feedback_dataset.csv'

# The dataset is already in the correct format for your pipeline
# with columns: feedback_id, customer_id, timestamp, rating, feedback_text, category
```

### 2. Running the Complete Pipeline

You can now run the complete feedback analysis pipeline:

```bash
# Run the full pipeline
python run_pipeline.py

# Or run individual components
python -m src.ingestor.ingest_feedback
python -m src.embedder.generate_embeddings
python -m src.clusterer.cluster_feedback
python -m src.topic_labeler.label_topics
python -m src.analysis.trend_detection
python -m src.analysis.cohort_analysis
```

### 3. Dashboard Analysis

Start the Streamlit dashboard to explore the data:

```bash
python run_dashboard.py
```

The dashboard will show:
- **Overview**: Key metrics and NPS score trends
- **Topic Analysis**: Clustering and topic insights
- **Trend Analysis**: Growing/declining topics and NPS changes
- **Cohort Analysis**: Customer behavior by signup cohorts
- **RAG Q&A**: Ask questions about the feedback data

## Dataset Advantages

### 1. Realistic NPS Distribution
- Follows actual NPS patterns with more promoters than detractors
- Includes realistic rating distribution across 0-10 scale
- Accounts for customer lifecycle and product evolution

### 2. Rich Metadata
- **Premium vs Regular**: Analyze differences between customer tiers
- **Spam Detection**: Filter out low-quality feedback
- **Temporal Data**: Track changes over time
- **Customer Journey**: Link feedback to signup dates

### 3. Synthetic Feedback Text
- Generated based on rating and customer attributes
- Realistic language patterns for each NPS category
- Consistent with rating scores
- Includes premium-specific feedback nuances

### 4. Comprehensive Categories
- Product Quality, Customer Service, Features, Value
- Technical Issues, Support, Performance, Pricing
- Premium-specific categories for enhanced analysis

## Analysis Opportunities

### 1. NPS Trend Analysis
- Track NPS changes over time
- Identify seasonal patterns
- Correlate with product releases or events

### 2. Cohort Analysis
- Compare customer behavior by signup month
- Analyze retention and satisfaction patterns
- Identify successful onboarding periods

### 3. Premium vs Regular Analysis
- Compare satisfaction levels between tiers
- Analyze premium feature impact
- Identify upselling opportunities

### 4. Topic Evolution
- Track how feedback topics change over time
- Identify emerging issues or successes
- Monitor product improvement impact

### 5. Predictive Analytics
- Predict customer churn based on feedback patterns
- Identify at-risk premium customers
- Forecast NPS trends

## Data Quality Notes

### Strengths
- Large sample size (1.5M+ records)
- Realistic temporal distribution
- Balanced representation of customer types
- Consistent data format

### Considerations
- Feedback text is synthetic (generated based on ratings)
- Categories are algorithmically assigned
- Spam accounts always give 0 ratings
- Premium customers tend to give slightly higher scores

## Next Steps

1. **Run Initial Analysis**: Use the existing pipeline to process this dataset
2. **Explore Dashboard**: Review all sections for insights
3. **Customize Categories**: Modify category generation if needed
4. **Add Real Data**: Replace with actual customer feedback when available
5. **Extend Analysis**: Add new analysis modules based on insights

## Support

If you encounter any issues with the dataset:
1. Check the data format matches your pipeline expectations
2. Verify all required columns are present
3. Review the processing script: `scripts/prepare_nps_dataset.py`
4. Regenerate the dataset if needed with different parameters

---

**Dataset prepared on**: $(date)
**Total processing time**: ~30 seconds
**Ready for analysis**: ✅