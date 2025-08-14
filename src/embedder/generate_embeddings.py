import os
import pandas as pd
import numpy as np
import logging
from utils import load_feedback_data, save_feedback_data, get_env_var, get_api_provider, get_embedding_client, get_model_name, get_embedding_provider

logger = logging.getLogger(__name__)

def get_embeddings(texts, model_name=None):
    """
    Generate embeddings for a list of texts using the configured embedding provider
    """
    embedding_provider = get_embedding_provider()
    
    if model_name is None:
        model_name = get_model_name('embedding')
    
    # Generate embeddings in batches
    batch_size = 100 if embedding_provider == 'gemini' else 1000  # Adjust based on API limits
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        logger.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Ensure batch_texts contains strings
        batch_texts = [str(text) for text in batch_texts]
        
        # Debug: Log what we're sending to the API
        logger.info(f"Sample batch_texts (first 2): {batch_texts[:2]}")
        logger.info(f"Type of first item: {type(batch_texts[0])}")
        
        try:
            if embedding_provider == 'openai':
                client = get_embedding_client()
                response = client.embeddings.create(
                    input=batch_texts,
                    model=model_name
                )
                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)
                
            elif embedding_provider == 'mistral':
                # Mistral uses direct HTTP API calls since get_embedding_client() returns None
                import requests
                import json
                
                base_url = get_env_var("MISTRAL_BASE_URL")
                api_key = get_env_var("MISTRAL_API_KEY")
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": model_name,
                    "input": batch_texts
                }
                
                response = requests.post(
                    f"{base_url}/embeddings",
                    headers=headers,
                    data=json.dumps(data)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    all_embeddings.extend(embeddings)
                else:
                    logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                    # Add empty embeddings for failed batch
                    embedding_dim = 1024  # Mistral embedding dimension
                    all_embeddings.extend([np.zeros(embedding_dim) for _ in range(len(batch_texts))])
                
            elif embedding_provider == 'gemini':
                import google.generativeai as genai
                client = get_embedding_client()  # This configures genai
                
                for text in batch_texts:
                    result = genai.embed_content(
                        model=model_name,
                        content=text,
                        task_type="retrieval_document"
                    )
                    all_embeddings.append(result['embedding'])
            
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
            # Add empty embeddings for failed batches
            embedding_dim = 1536 if get_api_provider() in ['openai', 'openrouter'] else 768
            all_embeddings.extend([np.zeros(embedding_dim) for _ in range(len(batch_texts))])
    
    return all_embeddings

def generate_and_save_embeddings(input_file, output_file):
    """
    Generate embeddings for feedback text and save to file
    """
    # Load feedback data
    df = load_feedback_data(input_file)
    
    # Get feedback texts
    texts = df['feedback_text'].tolist()
    
    # Debug: Check what we're getting from the DataFrame
    logger.info(f"First 3 texts from DataFrame: {texts[:3]}")
    logger.info(f"Type of first text: {type(texts[0])}")
    
    # Generate embeddings
    model_name = get_model_name('embedding')
    logger.info(f"Generating embeddings using model: {model_name}")
    embeddings = get_embeddings(texts)
    
    # Add embeddings to DataFrame
    df['embedding'] = embeddings
    
    # Save DataFrame with embeddings
    save_feedback_data(df, output_file)
    logger.info(f"Saved feedback with embeddings to {output_file}")
    
    return df