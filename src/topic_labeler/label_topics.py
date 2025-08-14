import argparse
import pandas as pd
import numpy as np
import os
import sys
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_env_var, load_feedback_data, save_feedback_data, get_api_provider, get_llm_client

logger = logging.getLogger(__name__)

def extract_keywords_tfidf(texts, n_keywords=5):
    """
    Extract keywords from a list of texts using TF-IDF
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_df=0.9,
        min_df=2,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top keywords for each text
    keywords = []
    for i in range(len(texts)):
        # Get the TF-IDF scores for this text
        tfidf_scores = tfidf_matrix[i].toarray()[0]
        
        # Get the indices of the top n_keywords scores
        top_indices = tfidf_scores.argsort()[-n_keywords:][::-1]
        
        # Get the corresponding feature names
        top_keywords = [feature_names[idx] for idx in top_indices]
        
        keywords.append(top_keywords)
    
    return keywords

def generate_cluster_keywords(df, cluster_col='cluster'):
    """
    Generate keywords for each cluster using TF-IDF
    """
    # Get unique clusters, excluding noise points (-1)
    clusters = sorted([c for c in df[cluster_col].unique() if c != -1])
    
    # Generate keywords for each cluster
    cluster_keywords = {}
    for cluster in clusters:
        # Get texts for this cluster
        cluster_texts = df[df[cluster_col] == cluster]['feedback_text'].tolist()
        
        # Skip if cluster is empty
        if not cluster_texts:
            continue
        
        # Extract keywords
        try:
            all_keywords = extract_keywords_tfidf(cluster_texts, n_keywords=10)
            
            # Flatten and count keywords
            keyword_counter = Counter([kw for keywords in all_keywords for kw in keywords])
            
            # Get top 5 keywords
            top_keywords = [kw for kw, _ in keyword_counter.most_common(5)]
            
            cluster_keywords[cluster] = top_keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords for cluster {cluster}: {e}")
            cluster_keywords[cluster] = []
    
    return cluster_keywords

def generate_topic_labels_llm(cluster_keywords, cluster_texts):
    """
    Generate descriptive topic labels for clusters using LLM
    """
    api_provider = get_api_provider()
    try:
        llm_model = get_env_var("LLM_MODEL")
    except ValueError:
        # Use default model based on API provider
        if api_provider in ["openai", "openrouter"]:
            llm_model = "gpt-3.5-turbo"
        else:
            llm_model = "gemini-1.5-flash"
    
    topic_labels = {}
    
    for cluster, keywords in cluster_keywords.items():
        # Get sample texts for this cluster (up to 5)
        texts = cluster_texts.get(cluster, [])
        sample_texts = texts[:5] if texts else []
        
        # Skip if no keywords or texts
        if not keywords or not sample_texts:
            topic_labels[cluster] = f"Cluster {cluster}"
            continue
        
        # Create prompt for LLM
        prompt = f"""Based on the following customer feedback and keywords, generate a short, descriptive topic label (3-5 words max).

Keywords: {', '.join(keywords)}

Sample feedback:
"""
        
        for i, text in enumerate(sample_texts, 1):
            prompt += f"\n{i}. {text}"
        
        prompt += "\n\nTopic label (3-5 words max):"
        
        try:
            # Generate topic label using LLM
            if api_provider in ["openai", "openrouter"]:
                client = get_llm_client()
                
                # Add extra headers for OpenRouter
                extra_kwargs = {}
                if api_provider == "openrouter":
                    extra_kwargs["extra_headers"] = {
                        "HTTP-Referer": get_env_var("OPENROUTER_SITE_URL") if os.getenv("OPENROUTER_SITE_URL") and os.getenv("OPENROUTER_SITE_URL") != "<YOUR_SITE_URL>" else "",
                        "X-Title": get_env_var("OPENROUTER_SITE_NAME") if os.getenv("OPENROUTER_SITE_NAME") and os.getenv("OPENROUTER_SITE_NAME") != "<YOUR_SITE_NAME>" else ""
                    }
                
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates concise, descriptive topic labels for customer feedback clusters."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=20,
                    temperature=0.3,
                    **extra_kwargs
                )
                topic_label = response.choices[0].message.content.strip()
            elif api_provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=get_env_var("GEMINI_API_KEY"))
                model = genai.GenerativeModel(llm_model)
                full_prompt = f"You are a helpful assistant that generates concise, descriptive topic labels for customer feedback clusters.\n\n{prompt}"
                response = model.generate_content(full_prompt)
                topic_label = response.text.strip()
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
            
            # Topic label is already extracted in the if-elif block above
            
            # Truncate if too long
            if len(topic_label.split()) > 5:
                topic_label = ' '.join(topic_label.split()[:5])
            
            topic_labels[cluster] = topic_label
            
        except Exception as e:
            logger.error(f"Error generating topic label for cluster {cluster}: {e}")
            topic_labels[cluster] = f"Cluster {cluster}"
    
    return topic_labels

def label_topics(input_file, output_file=None):
    """
    Label topics for clusters in feedback data
    """
    # Load data with clusters
    df = load_feedback_data(input_file)
    if df is None:
        return None
    
    # Check if clusters are present
    if 'cluster' not in df.columns:
        logger.error(f"No cluster information found in {input_file}")
        return None
    
    # Generate keywords for each cluster
    logger.info("Generating keywords for clusters")
    cluster_keywords = generate_cluster_keywords(df)
    
    # Get texts for each cluster
    cluster_texts = {}
    for cluster in cluster_keywords.keys():
        cluster_texts[cluster] = df[df['cluster'] == cluster]['feedback_text'].tolist()
    
    # Generate topic labels using LLM
    logger.info("Generating topic labels using LLM")
    topic_labels = generate_topic_labels_llm(cluster_keywords, cluster_texts)
    
    # Add topic labels to dataframe
    df['topic'] = df['cluster'].map(lambda x: topic_labels.get(x, f"Cluster {x}") if x != -1 else "Noise")
    
    # Save dataframe with topic labels if output file is specified
    if output_file:
        save_feedback_data(df, output_file)
        logger.info(f"Saved feedback with topic labels to {output_file}")
    
    # Create a summary dataframe with cluster information
    summary_data = []
    for cluster, label in topic_labels.items():
        cluster_size = len(df[df['cluster'] == cluster])
        avg_nps = df[df['cluster'] == cluster]['nps_score'].mean()
        keywords = ', '.join(cluster_keywords.get(cluster, []))
        
        summary_data.append({
            'cluster': cluster,
            'topic': label,
            'size': cluster_size,
            'avg_nps': avg_nps,
            'keywords': keywords
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary if output file is specified
    if output_file:
        summary_path = os.path.join(
            os.path.dirname(output_file),
            f"topic_summary_{os.path.basename(output_file)}"
        )
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved topic summary to {summary_path}")
    
    return df, summary_df

def main():
    parser = argparse.ArgumentParser(description='Label topics for customer feedback clusters')
    parser.add_argument('--input', required=True, help='Input CSV file with clustered feedback')
    parser.add_argument('--output', help='Output CSV file for feedback with topic labels')
    args = parser.parse_args()
    
    # If output is not specified, use a default name based on the input file
    if not args.output:
        input_dir = os.path.dirname(args.input)
        input_filename = os.path.basename(args.input)
        output_filename = f"labeled_{input_filename}"
        args.output = os.path.join(input_dir, output_filename)
    
    # Label topics
    result = label_topics(args.input, args.output)
    
    if result is not None:
        df, summary_df = result
        
        # Print summary
        print(f"\nTopic Labeling Summary:")
        print(f"Total entries: {len(df)}")
        print(f"Number of topics: {len(summary_df)}")
        print(f"\nTopics:")
        for _, row in summary_df.iterrows():
            print(f"  Cluster {row['cluster']} ({row['size']} entries, Avg NPS: {row['avg_nps']:.2f}): {row['topic']}")
            print(f"    Keywords: {row['keywords']}")
        
        print(f"\nLabeled data saved to: {args.output}")

if __name__ == "__main__":
    main()