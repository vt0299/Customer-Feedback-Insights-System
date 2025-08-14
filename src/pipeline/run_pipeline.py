import argparse
import os
import sys
import logging
import time

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_env_var
from ingestor.ingest_feedback import ingest_feedback
from embedder.generate_embeddings import generate_and_save_embeddings
from clusterer.cluster_feedback import cluster_feedback
from topic_labeler.label_topics import label_topics
from rag.rag_answerer import create_rag_system
from evaluation.evaluate_rag import evaluate_rag_system, create_test_queries, load_vector_store

logger = logging.getLogger(__name__)

def run_pipeline(input_file, output_dir=None, skip_steps=None, n_eval_queries=10):
    """
    Run the complete feedback analysis pipeline
    """
    start_time = time.time()
    
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default skip_steps if not specified
    if skip_steps is None:
        skip_steps = []
    
    # Step 1: Ingest and clean feedback
    if 'ingest' not in skip_steps:
        logger.info("Step 1: Ingesting and cleaning feedback data")
        cleaned_path = os.path.join(output_dir, 'cleaned_feedback.csv')
        df_cleaned = ingest_feedback(input_file, cleaned_path)
        if df_cleaned is None:
            logger.error("Failed to ingest feedback data. Exiting pipeline.")
            return False
        logger.info(f"Cleaned feedback saved to {cleaned_path}")
    else:
        logger.info("Skipping ingestion step")
        cleaned_path = os.path.join(output_dir, 'cleaned_feedback.csv')
        if not os.path.exists(cleaned_path):
            logger.error(f"Cleaned feedback file not found at {cleaned_path}. Cannot proceed.")
            return False
    
    # Step 2: Generate embeddings
    if 'embed' not in skip_steps:
        logger.info("Step 2: Generating embeddings for feedback text")
        embedded_path = os.path.join(output_dir, 'embedded_feedback.csv')
        df_embedded = generate_and_save_embeddings(cleaned_path, embedded_path)
        if df_embedded is None:
            logger.error("Failed to generate embeddings. Exiting pipeline.")
            return False
        logger.info(f"Feedback with embeddings saved to {embedded_path}")
    else:
        logger.info("Skipping embedding step")
        embedded_path = os.path.join(output_dir, 'embedded_feedback.csv')
        if not os.path.exists(embedded_path):
            logger.error(f"Embedded feedback file not found at {embedded_path}. Cannot proceed.")
            return False
    
    # Step 3: Cluster feedback
    if 'cluster' not in skip_steps:
        logger.info("Step 3: Clustering feedback")
        clustered_path = os.path.join(output_dir, 'clustered_feedback.csv')
        df_clustered = cluster_feedback(
            embedded_path, 
            clustered_path, 
            visualize=True
        )
        if df_clustered is None:
            logger.error("Failed to cluster feedback. Exiting pipeline.")
            return False
        logger.info(f"Clustered feedback saved to {clustered_path}")
    else:
        logger.info("Skipping clustering step")
        clustered_path = os.path.join(output_dir, 'clustered_feedback.csv')
        if not os.path.exists(clustered_path):
            logger.error(f"Clustered feedback file not found at {clustered_path}. Cannot proceed.")
            return False
    
    # Step 4: Label topics
    if 'label' not in skip_steps:
        logger.info("Step 4: Labeling topics for clusters")
        labeled_path = os.path.join(output_dir, 'labeled_feedback.csv')
        result = label_topics(clustered_path, labeled_path)
        if result is None:
            logger.error("Failed to label topics. Exiting pipeline.")
            return False
        df_labeled, summary_df = result
        logger.info(f"Feedback with topic labels saved to {labeled_path}")
    else:
        logger.info("Skipping topic labeling step")
        labeled_path = os.path.join(output_dir, 'labeled_feedback.csv')
        if not os.path.exists(labeled_path):
            logger.error(f"Labeled feedback file not found at {labeled_path}. Cannot proceed.")
            return False
    
    # Step 5: Create RAG system
    if 'rag' not in skip_steps:
        logger.info("Step 5: Creating RAG system")
        rag_output_dir = os.path.join(output_dir, 'rag_output')
        vector_store = create_rag_system(labeled_path, rag_output_dir)
        if vector_store is None:
            logger.error("Failed to create RAG system. Exiting pipeline.")
            return False
        logger.info(f"RAG system created in {rag_output_dir}")
    else:
        logger.info("Skipping RAG system creation step")
        rag_output_dir = os.path.join(output_dir, 'rag_output')
        vector_store_dir = os.path.join(rag_output_dir, 'vector_store')
        if not os.path.exists(vector_store_dir):
            logger.error(f"Vector store not found at {vector_store_dir}. Cannot proceed with evaluation.")
            vector_store = None
        else:
            vector_store = load_vector_store(vector_store_dir)
    
    # Step 6: Evaluate RAG system
    if 'evaluate' not in skip_steps and vector_store is not None:
        logger.info("Step 6: Evaluating RAG system")
        
        # Load labeled feedback data
        import pandas as pd
        df_labeled = pd.read_csv(labeled_path)
        
        # Create test queries
        test_queries = create_test_queries(df_labeled, n_eval_queries)
        
        # Evaluate RAG system
        eval_output_path = os.path.join(rag_output_dir, 'evaluation_results.json')
        results, evaluation_metrics = evaluate_rag_system(vector_store, test_queries)
        
        # Extract average scores from evaluation metrics
        avg_groundedness = evaluation_metrics.get('avg_groundedness', 0)
        avg_relevance = evaluation_metrics.get('avg_relevance', 0)
        avg_correctness = 0  # Not implemented in current evaluation system
        
        logger.info(f"Evaluation complete. Average scores:")
        logger.info(f"  Groundedness: {avg_groundedness:.2f}/1.0")
        logger.info(f"  Relevance: {avg_relevance:.2f}/1.0")
        logger.info(f"  Correctness: {avg_correctness:.2f}/1.0")
        logger.info(f"  Overall: {(avg_groundedness + avg_relevance + avg_correctness) / 3:.2f}/1.0")
    else:
        logger.info("Skipping evaluation step")
    
    # Calculate total runtime
    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"Pipeline completed in {runtime:.2f} seconds")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run the complete feedback analysis pipeline')
    parser.add_argument('--input', help='Input CSV file with raw feedback data')
    parser.add_argument('--output-dir', help='Output directory for all pipeline artifacts')
    parser.add_argument('--skip', nargs='+', choices=['ingest', 'embed', 'cluster', 'label', 'rag', 'evaluate'],
                        help='Steps to skip in the pipeline')
    parser.add_argument('--n-eval-queries', type=int, default=10, help='Number of queries to use for evaluation')
    args = parser.parse_args()
    
    # If input is not specified, use the default sample data
    if not args.input:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir))
        args.input = os.path.join(project_dir, 'data', 'sample_feedback.csv')
    
    # If output directory is not specified, use the data directory
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.input)
    
    # Run the pipeline
    success = run_pipeline(
        args.input,
        args.output_dir,
        args.skip,
        args.n_eval_queries
    )
    
    if success:
        print("\nPipeline completed successfully!")
        print(f"Output directory: {args.output_dir}")
        print("\nTo start the dashboard, run:")
        print("python -m streamlit run src/dashboard/app.py")
    else:
        print("\nPipeline failed. Check the logs for details.")

if __name__ == "__main__":
    main()