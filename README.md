# Customer Feedback Insights System

A complete end-to-end system for analyzing customer feedback using RAG (Retrieval Augmented Generation), clustering, and interactive dashboards.

## Architecture

```
feedback-ingestor → embedder → clusterer (HDBSCAN/KMeans) → topic-labeler → RAG-answerer → dashboard
```

## Features

- Ingest NPS/free-text feedback
- Cluster feedback themes with embeddings
- Label topics automatically
- RAG over feedback for "why" explanations
- Interactive dashboard (Streamlit)
- Evaluation system for summaries (TruLens)

## Setup Instructions

### Prerequisites

- Python 3.9+
- API key from one of the supported providers:
  - OpenAI API key
  - Gemini API key
  - OpenRouter API key (supports multiple models including free options)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your API configuration:
   
   **For OpenAI:**
   ```
   API_PROVIDER=openai
   OPENAI_API_KEY=your_api_key_here
   ```
   
   **For Gemini:**
   ```
   API_PROVIDER=gemini
   GEMINI_API_KEY=your_api_key_here
   ```
   
   **For OpenRouter (using OpenAI client format):**
   ```
   API_PROVIDER=openai
   # OpenAI API Key (required for embeddings - OpenRouter doesn't support embedding models)
   OPENAI_API_KEY=your_openai_api_key_here
   # Your OpenRouter API Key (used for LLM via OpenAI client format)
   OPENROUTER_API_KEY_FOR_LLM=your_openrouter_api_key_here
   EMBEDDING_MODEL=text-embedding-ada-002
   LLM_MODEL=openai/gpt-oss-20b:free
   ```
   
   **Note:** This configuration uses OpenRouter for LLM operations through the OpenAI client interface, while using OpenAI for embeddings (since OpenRouter doesn't support embedding models).

## Running the System

### 1. Data Ingestion

```bash
python src/ingestor/ingest_feedback.py --input data/sample_feedback.csv
```

### 2. Run the Complete Pipeline

```bash
python src/pipeline/run_pipeline.py
```

### 3. Start the Dashboard

```bash
python -m streamlit run src/dashboard/app.py
```

## Project Structure

```
.
├── data/                  # Data storage
├── src/                   # Source code
│   ├── ingestor/          # Data ingestion module
│   ├── embedder/          # Text embedding generation
│   ├── clusterer/         # Clustering algorithms
│   ├── topic_labeler/     # Topic labeling for clusters
│   ├── rag/               # RAG-based answering system
│   ├── evaluation/        # Evaluation with TruLens
│   ├── dashboard/         # Streamlit dashboard
│   └── pipeline/          # End-to-end pipeline
├── tests/                 # Unit and integration tests
├── .env                   # Environment variables
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Workflow

1. Define KPIs
2. Build pipeline
3. Evaluate summaries (TruLens/LLM-as-judge)
4. Ship
5. Monitor

## Stretch Goals

- Trend detection
- Cohort analysis
- Cost/latency budget alerts

## Output

[!Alt images](images/1.png)
[!Alt images](images/2.png)
[!Alt images](images/3.png)
[!Alt images](images/4.png)
[!Alt images](images/5.png)
[!Alt images](images/6.png)
