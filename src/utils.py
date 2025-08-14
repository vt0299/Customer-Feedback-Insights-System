import os
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_env_var(var_name):
    """
    Get environment variable or raise an error if not found
    """
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable {var_name} not found")
    return value

def load_feedback_data(file_path):
    """
    Load feedback data from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert embedding strings back to numpy arrays if present
        if 'embedding' in df.columns:
            def parse_embedding_string(x):
                if isinstance(x, str) and x.strip():
                    try:
                        # Remove brackets and quotes
                        clean_str = x.strip('[]"')
                        # Split by comma and strip whitespace
                        values = [float(val.strip()) for val in clean_str.split(',') if val.strip()]
                        return np.array(values)
                    except Exception as e:
                        logger.warning(f"Failed to parse embedding: {e}")
                        return np.array([])
                return x
            
            df['embedding'] = df['embedding'].apply(parse_embedding_string)
        
        logger.info(f"Loaded {len(df)} feedback entries from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading feedback data from {file_path}: {e}")
        return None

def save_feedback_data(df, file_path):
    """
    Save feedback data to CSV file
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Convert embeddings to string representation for saving
    if 'embedding' in df_copy.columns:
        df_copy['embedding'] = df_copy['embedding'].apply(
            lambda x: ' '.join(map(str, x)) if isinstance(x, np.ndarray) else x
        )
    
    # Save to CSV with proper quoting to handle embedding strings
    df_copy.to_csv(file_path, index=False, quoting=1)  # QUOTE_ALL
    logger.info(f"Saved {len(df_copy)} feedback entries to {file_path}")

def get_api_provider():
    """
    Get the configured API provider (openai, gemini, or openrouter)
    """
    provider = os.getenv('API_PROVIDER', 'openai').lower()
    if provider not in ['openai', 'gemini', 'openrouter']:
        raise ValueError(f"Unsupported API provider: {provider}. Must be 'openai', 'gemini', or 'openrouter'")
    return provider

def get_embedding_provider():
    """
    Get the embedding provider from environment variables
    """
    try:
        return get_env_var("EMBEDDING_PROVIDER").lower()
    except ValueError:
        # Default to same as API provider if not specified
        return get_api_provider()

def get_llm_client():
    """
    Get the appropriate LLM client based on API provider
    """
    api_provider = get_api_provider()
    
    if api_provider == 'openai':
        from openai import OpenAI
        # Check if we have a custom base URL (for services like Chutes AI)
        try:
            base_url = get_env_var("OPENAI_BASE_URL")
            return OpenAI(
                base_url=base_url,
                api_key=get_env_var("OPENAI_API_KEY")
            )
        except ValueError:
            # Standard OpenAI configuration (no custom base URL)
            return OpenAI(api_key=get_env_var("OPENAI_API_KEY"))
    elif api_provider == 'openrouter':
        from openai import OpenAI
        return OpenAI(
            base_url=get_env_var("OPENROUTER_BASE_URL"),
            api_key=get_env_var("OPENROUTER_API_KEY")
        )
    elif api_provider == 'gemini':
        import google.generativeai as genai
        genai.configure(api_key=get_env_var("GEMINI_API_KEY"))
        return genai
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")

def get_embedding_client():
    """
    Get the appropriate embedding client based on embedding provider
    """
    embedding_provider = get_embedding_provider()
    
    if embedding_provider == 'openai':
        from openai import OpenAI
        try:
            # Try to use dedicated embedding API key first
            api_key = get_env_var("OPENAI_EMBEDDING_API_KEY")
            return OpenAI(api_key=api_key)
        except ValueError:
            # Fall back to OpenRouter API key and base URL
            api_key = get_env_var("OPENAI_API_KEY")
            try:
                base_url = get_env_var("OPENROUTER_BASE_URL")
                return OpenAI(api_key=api_key, base_url=base_url)
            except ValueError:
                return OpenAI(api_key=api_key)
    elif embedding_provider == 'mistral':
        # Return None for Mistral to use direct HTTP API call in generate_embeddings.py
        # This prevents automatic tokenization by OpenAI client
        return None
    elif embedding_provider == 'gemini':
        import google.generativeai as genai
        genai.configure(api_key=get_env_var("GEMINI_API_KEY"))
        return genai
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")

def get_model_name(model_type='llm'):
    """
    Get the model name based on type and provider
    """
    if model_type == 'llm':
        return get_env_var("LLM_MODEL")
    elif model_type == 'embedding':
        return get_env_var("EMBEDDING_MODEL")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_embedding_dimensions():
    """
    Get the embedding dimensions from environment variables
    """
    try:
        return int(get_env_var("EMBEDDING_DIM"))
    except ValueError:
        # Default dimensions for common models
        model = get_model_name('embedding')
        if 'ada-002' in model:
            return 1536
        elif 'mistral-embed' in model:
            return 1024
        else:
            return 1536  # Default fallback

def get_clustering_config():
    """
    Get clustering configuration from environment variables
    """
    return {
        'algorithm': os.getenv('CLUSTERING_ALGORITHM', 'hdbscan'),
        'n_clusters': int(os.getenv('N_CLUSTERS', 10))
    }