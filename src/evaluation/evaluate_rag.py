import argparse
import pandas as pd
import os
import sys
import json
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_env_var, load_feedback_data
from rag.rag_answerer import load_vector_store, generate_rag_answer

logger = logging.getLogger(__name__)

def create_test_queries(df, n_queries=10):
    """
    Create test queries from feedback data
    """
    # Get unique topics, excluding noise
    topics = df[df['cluster'] != -1]['topic'].unique()
    
    # Create queries for each topic
    queries = []
    for topic in topics[:min(5, len(topics))]:
        queries.append(f"What are the main issues mentioned in the '{topic}' feedback?")
        queries.append(f"What positive aspects are highlighted in the '{topic}' feedback?")
    
    # Add general queries
    general_queries = [
        "What are the top 3 issues customers are facing?",
        "What features do customers like the most?",
        "What improvements would have the biggest impact on customer satisfaction?",
        "Why are customers giving low NPS scores?",
        "What's driving the high NPS scores?"
    ]
    
    queries.extend(general_queries)
    
    # Return the requested number of queries
    return queries[:n_queries]

def evaluate_groundedness(answer, context):
    """
    Evaluate if the answer is grounded in the context
    """
    from utils import get_api_provider, get_llm_client
    
    prompt = f"""
    You are evaluating how well a given answer is grounded in the provided context.
    Please rate the groundedness on a scale of 0-1, where 1 means the answer is fully supported by the context.
    
    Context: {context}
    
    Answer: {answer}
    
    Provide only a numerical score between 0 and 1.
    """
    
    api_provider = get_api_provider()
    
    try:
        if api_provider in ["openai", "openrouter"]:
            client = get_llm_client()
            try:
                llm_model = get_env_var("LLM_MODEL")
            except ValueError:
                llm_model = "gpt-3.5-turbo"
            
            # Add extra headers for OpenRouter
            extra_kwargs = {}
            if api_provider == "openrouter":
                extra_kwargs["extra_headers"] = {
                    "HTTP-Referer": get_env_var("OPENROUTER_SITE_URL") if os.getenv("OPENROUTER_SITE_URL") and os.getenv("OPENROUTER_SITE_URL") != "<YOUR_SITE_URL>" else "",
                    "X-Title": get_env_var("OPENROUTER_SITE_NAME") if os.getenv("OPENROUTER_SITE_NAME") and os.getenv("OPENROUTER_SITE_NAME") != "<YOUR_SITE_NAME>" else ""
                }
            
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                **extra_kwargs
            )
            score = float(response.choices[0].message.content.strip())
        elif api_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=get_env_var("GEMINI_API_KEY"))
            try:
                llm_model = get_env_var("LLM_MODEL")
            except ValueError:
                llm_model = "gemini-1.5-flash"
            
            model = genai.GenerativeModel(llm_model)
            response = model.generate_content(prompt)
            score = float(response.text.strip())
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")
        
        return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
    except:
        return 0.5  # Default score if parsing fails

def evaluate_relevance(query, answer):
    """
    Evaluate how relevant the answer is to the query
    """
    from utils import get_api_provider, get_llm_client
    
    prompt = f"""
    You are evaluating how relevant a given answer is to a query.
    Please rate the relevance on a scale of 0-1, where 1 means the answer is highly relevant to the query.
    
    Query: {query}
    
    Answer: {answer}
    
    Provide only a numerical score between 0 and 1.
    """
    
    api_provider = get_api_provider()
    
    try:
        if api_provider in ["openai", "openrouter"]:
            client = get_llm_client()
            try:
                llm_model = get_env_var("LLM_MODEL")
            except ValueError:
                llm_model = "gpt-3.5-turbo"
            
            # Add extra headers for OpenRouter
            extra_kwargs = {}
            if api_provider == "openrouter":
                extra_kwargs["extra_headers"] = {
                    "HTTP-Referer": get_env_var("OPENROUTER_SITE_URL") if os.getenv("OPENROUTER_SITE_URL") and os.getenv("OPENROUTER_SITE_URL") != "<YOUR_SITE_URL>" else "",
                    "X-Title": get_env_var("OPENROUTER_SITE_NAME") if os.getenv("OPENROUTER_SITE_NAME") and os.getenv("OPENROUTER_SITE_NAME") != "<YOUR_SITE_NAME>" else ""
                }
            
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                **extra_kwargs
            )
            score = float(response.choices[0].message.content.strip())
        elif api_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=get_env_var("GEMINI_API_KEY"))
            try:
                llm_model = get_env_var("LLM_MODEL")
            except ValueError:
                 llm_model = "gemini-1.5-flash"
            
            model = genai.GenerativeModel(llm_model)
            response = model.generate_content(prompt)
            score = float(response.text.strip())
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")
        
        return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
    except:
        return 0.5  # Default score if parsing fails

def evaluate_rag_system(vector_store, test_queries):
    """
    Evaluate RAG system using custom evaluation functions
    """
    results = []
    groundedness_scores = []
    relevance_scores = []
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"Evaluating query {i}/{len(test_queries)}: {query}")
        
        # Generate RAG answer
        result = generate_rag_answer(query, vector_store)
        answer = result['answer']
        context = "\n".join([source['feedback_text'] for source in result['sources']])
        
        # Evaluate groundedness
        groundedness = evaluate_groundedness(answer, context)
        groundedness_scores.append(groundedness)
        
        # Evaluate relevance
        relevance = evaluate_relevance(query, answer)
        relevance_scores.append(relevance)
        
        # Store results
        results.append({
            'query': query,
            'response': answer,
            'context': context,
            'groundedness': groundedness,
            'relevance': relevance
        })
    
    # Calculate average scores
    avg_groundedness = sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else 0
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    avg_overall = (avg_groundedness + avg_relevance) / 2
    
    return results, {
        'avg_groundedness': avg_groundedness,
        'avg_relevance': avg_relevance,
        'avg_overall': avg_overall
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system using custom evaluation')
    parser.add_argument('--vector-store', required=True, help='Directory containing the vector store')
    parser.add_argument('--feedback-data', required=True, help='CSV file with labeled feedback data')
    parser.add_argument('--output', help='Output JSON file for evaluation results')
    parser.add_argument('--n-queries', type=int, default=10, help='Number of test queries to evaluate')
    args = parser.parse_args()
    
    # If output is not specified, use a default name
    if not args.output:
        args.output = os.path.join(os.path.dirname(args.vector_store), 'evaluation_results.json')
    
    # Load vector store
    vector_store = load_vector_store(args.vector_store)
    
    # Load feedback data
    df = load_feedback_data(args.feedback_data)
    
    # Create test queries
    test_queries = create_test_queries(df, args.n_queries)
    
    # Evaluate RAG system
    results, metrics = evaluate_rag_system(vector_store, test_queries)
    
    # Create evaluation summary
    evaluation_summary = {
        'metrics': {
            'groundedness': metrics['avg_groundedness'],
            'relevance': metrics['avg_relevance'],
            'overall': metrics['avg_overall']
        },
        'results': results
    }
    
    # Save evaluation results
    with open(args.output, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    # Print summary
    print(f"\nRAG Evaluation Summary:")
    print(f"Number of queries evaluated: {len(test_queries)}")
    print(f"\nAverage scores:")
    print(f"  Groundedness: {metrics['avg_groundedness']:.2f}/1.0")
    print(f"  Relevance: {metrics['avg_relevance']:.2f}/1.0")
    print(f"  Overall: {metrics['avg_overall']:.2f}/1.0")
    print(f"\nEvaluation results saved to: {args.output}")

if __name__ == "__main__":
    main()