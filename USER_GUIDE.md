# Customer Feedback Insights System - User Guide

This guide provides detailed instructions on how to set up and use the Customer Feedback Insights System.

## Table of Contents

1. [System Overview](#system-overview)
2. [Setup Instructions](#setup-instructions)
3. [Running the Pipeline](#running-the-pipeline)
4. [Using the Dashboard](#using-the-dashboard)
5. [Understanding the Results](#understanding-the-results)
6. [Troubleshooting](#troubleshooting)

## System Overview

The Customer Feedback Insights System is an end-to-end solution for analyzing customer feedback data. It processes raw feedback text and NPS scores to identify key themes, generate insights, and provide a question-answering system over your feedback data.

The system consists of the following components:

1. **Feedback Ingestor**: Cleans and preprocesses raw feedback data
2. **Embedder**: Generates vector embeddings for feedback text
3. **Clusterer**: Groups similar feedback using HDBSCAN or KMeans
4. **Topic Labeler**: Automatically labels clusters with descriptive topics
5. **RAG Answerer**: Provides a retrieval-augmented generation system for answering questions about feedback
6. **Dashboard**: Visualizes insights and provides an interface for exploring feedback

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- API key from one of the supported providers:
  - OpenAI API key
  - Gemini API key  
  - OpenRouter API key (supports multiple models including free options)

### Installation

1. Clone or download the project to your local machine

2. Run the setup script to install dependencies and create necessary files:

   ```bash
   python setup.py
   ```

3. Edit the `.env` file to add your API configuration:

   **For OpenAI:**
   ```
   API_PROVIDER=openai
   OPENAI_API_KEY=your_api_key_here
   EMBEDDING_MODEL=text-embedding-3-small
   LLM_MODEL=gpt-3.5-turbo
   CLUSTERING_ALGORITHM=hdbscan  # or kmeans
   ```

   **For Gemini:**
   ```
   API_PROVIDER=gemini
   GEMINI_API_KEY=your_api_key_here
   EMBEDDING_MODEL=models/embedding-001
   LLM_MODEL=gemini-pro
   CLUSTERING_ALGORITHM=hdbscan  # or kmeans
   ```

   **For OpenRouter:**
   ```
   API_PROVIDER=openrouter
   OPENROUTER_API_KEY=your_api_key_here
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   OPENROUTER_SITE_URL=<YOUR_SITE_URL>
   OPENROUTER_SITE_NAME=<YOUR_SITE_NAME>
   # OpenAI API Key (required for embeddings - OpenRouter doesn't support embedding models)
# You need a separate OpenAI API key here, not your OpenRouter key
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE

# Your OpenRouter API Key (used for LLM via OpenAI client format)
OPENROUTER_API_KEY_FOR_LLM=your_openrouter_api_key_here
   EMBEDDING_MODEL=text-embedding-3-small
   LLM_MODEL=google/gemma-3-4b-it:free
   CLUSTERING_ALGORITHM=hdbscan  # or kmeans
   N_CLUSTERS=10  # for kmeans
   MIN_CLUSTER_SIZE=5  # for hdbscan
   DASHBOARD_PORT=8501
   ENABLE_EVALUATION=true
   ```

## Running the Pipeline

### Using Sample Data

To run the pipeline with the included sample data:

```bash
python run_pipeline.py
```

This will process the sample feedback data and generate all necessary outputs in the `data` directory.

### Using Your Own Data

To use your own feedback data, prepare a CSV file with at least the following columns:
- `feedback_id`: Unique identifier for each feedback
- `date`: Date of feedback (in any standard format)
- `nps_score`: Numerical NPS score (0-10)
- `feedback_text`: The text of the customer feedback

Then run the pipeline with your data file:

```bash
python run_pipeline.py --input /path/to/your/feedback_data.csv --output-dir /path/to/output/directory
```

### Pipeline Options

The pipeline script supports several options:

- `--input`: Path to input CSV file (defaults to sample data)
- `--output-dir`: Directory to store output files (defaults to same directory as input)
- `--skip`: Steps to skip in the pipeline (e.g., `--skip ingest embed` to skip ingestion and embedding)
- `--n-eval-queries`: Number of queries to use for evaluation (default: 10)

Example with options:

```bash
python run_pipeline.py --input data/my_feedback.csv --output-dir data/results --skip evaluate
```

## Using the Dashboard

To start the dashboard:

```bash
python run_dashboard.py
```

This will launch a Streamlit dashboard in your default web browser (typically at http://localhost:8501).

### Dashboard Features

The dashboard provides several views:

1. **Overview**: Summary statistics of your feedback data
2. **Topic Insights**: Visualization of identified topics and their characteristics
3. **Feedback Explorer**: Interactive exploration of feedback by cluster, NPS category, or search
4. **RAG Q&A**: Ask questions about your feedback data and get AI-generated answers
5. **Evaluation**: View evaluation results of the RAG system (if evaluation was enabled)

## Understanding the Results

### Output Files

The pipeline generates several output files:

- `cleaned_feedback.csv`: Preprocessed feedback data
- `embedded_feedback.csv`: Feedback with vector embeddings
- `clustered_feedback.csv`: Feedback with cluster assignments
- `labeled_feedback.csv`: Feedback with topic labels
- `rag_output/`: Directory containing RAG system files
  - `vector_store/`: Chroma vector database
  - `topic_insights.json`: Generated insights for each topic
  - `evaluation_results.json`: Evaluation results (if enabled)

### Visualizations

The clustering step generates several visualizations in the output directory:

- `cluster_visualization.png`: 2D projection of feedback clusters
- `cluster_sizes.png`: Distribution of cluster sizes
- `nps_distribution.png`: NPS distribution by cluster

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure your OpenAI API key is correctly set in the `.env` file
   - Check that your API key has sufficient quota

2. **Memory Issues**:
   - For large datasets, you may need to increase your system's available memory
   - Consider processing data in batches by modifying the pipeline code

3. **Clustering Quality**:
   - If using KMeans, try different values for `N_CLUSTERS`
   - If using HDBSCAN, adjust `MIN_CLUSTER_SIZE` to get better clusters

### Logs

The system logs detailed information during execution. Check the console output for warnings and errors.

### Getting Help

If you encounter issues not covered in this guide, please check the project's README.md file for additional information or contact the project maintainers.