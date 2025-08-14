import argparse
import pandas as pd
import numpy as np
import os
import sys
import logging
import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_env_var, load_feedback_data, save_feedback_data, get_api_provider, get_llm_client, get_embedding_client, get_embedding_provider, get_model_name
from embedder.generate_embeddings import get_embeddings

logger = logging.getLogger(__name__)

def create_vector_store(df, persist_directory=None):
    """
    Create a vector store from feedback data
    """
    # Create documents from feedback
    documents = []
    for _, row in df.iterrows():
        # Create metadata
        metadata = {
            'feedback_id': row['feedback_id'],
            'date': str(row['date']),
            'nps_score': int(row['nps_score']),
            'nps_category': row['nps_category'],
            'cluster': int(row['cluster']) if row['cluster'] != -1 else -1,
            'topic': row['topic'] if 'topic' in row else ''
        }
        
        # Create document
        doc = Document(
            page_content=row['feedback_text'],
            metadata=metadata
        )
        
        documents.append(doc)
    
    # Create vector store with appropriate embedding function
    embedding_provider = get_embedding_provider()
    embedding_model = get_model_name('embedding')
    
    if embedding_provider == "openai":
        try:
            # Try to use dedicated embedding API key first
            api_key = get_env_var("OPENAI_EMBEDDING_API_KEY")
        except ValueError:
            # Fall back to main OpenAI API key
            api_key = get_env_var("OPENAI_API_KEY")
        
        embedding_function = OpenAIEmbeddings(
            api_key=api_key,
            model=embedding_model
        )
    elif embedding_provider == "mistral":
        # Use custom embedding function for Mistral to avoid tokenization
        from langchain_core.embeddings import Embeddings
        import requests
        
        class MistralEmbeddings(Embeddings):
            def __init__(self, api_key, model, base_url):
                self.api_key = api_key
                self.model = model
                self.base_url = base_url
            
            def embed_documents(self, texts):
                return self._get_embeddings(texts)
            
            def embed_query(self, text):
                return self._get_embeddings([text])[0]
            
            def _get_embeddings(self, texts):
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                data = {
                    "model": self.model,
                    "input": texts
                }
                
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=data
                )
                
                if response.status_code != 200:
                    raise Exception(f"Mistral API error: {response.status_code} - {response.text}")
                
                result = response.json()
                return [item["embedding"] for item in result["data"]]
        
        embedding_function = MistralEmbeddings(
            api_key=get_env_var("MISTRAL_API_KEY"),
            model=embedding_model,
            base_url=get_env_var("MISTRAL_BASE_URL")
        )
    elif embedding_provider == "gemini":
        embedding_function = GoogleGenerativeAIEmbeddings(
            google_api_key=get_env_var("GEMINI_API_KEY"),
            model=embedding_model
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
    
    if persist_directory:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        vector_store.persist()
        logger.info(f"Created and persisted vector store with {len(documents)} documents")
    else:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function
        )
        logger.info(f"Created in-memory vector store with {len(documents)} documents")
    
    return vector_store

def load_vector_store(persist_directory):
    """
    Load a vector store from disk with async event loop handling
    """
    import asyncio
    import threading
    
    embedding_provider = get_embedding_provider()
    embedding_model = get_model_name('embedding')
    
    if embedding_provider == "openai":
        try:
            # Try to use dedicated embedding API key first
            api_key = get_env_var("OPENAI_EMBEDDING_API_KEY")
        except ValueError:
            # Fall back to main OpenAI API key
            api_key = get_env_var("OPENAI_API_KEY")
        
        embedding_function = OpenAIEmbeddings(
            api_key=api_key,
            model=embedding_model
        )
    elif embedding_provider == "mistral":
        # Use custom embedding function for Mistral to avoid tokenization
        from langchain_core.embeddings import Embeddings
        import requests
        
        class MistralEmbeddings(Embeddings):
            def __init__(self, api_key, model, base_url):
                self.api_key = api_key
                self.model = model
                self.base_url = base_url
            
            def embed_documents(self, texts):
                return self._get_embeddings(texts)
            
            def embed_query(self, text):
                return self._get_embeddings([text])[0]
            
            def _get_embeddings(self, texts):
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                data = {
                    "model": self.model,
                    "input": texts
                }
                
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=data
                )
                
                if response.status_code != 200:
                    raise Exception(f"Mistral API error: {response.status_code} - {response.text}")
                
                result = response.json()
                return [item["embedding"] for item in result["data"]]
        
        embedding_function = MistralEmbeddings(
            api_key=get_env_var("MISTRAL_API_KEY"),
            model=embedding_model,
            base_url=get_env_var("MISTRAL_BASE_URL")
        )
    elif embedding_provider == "gemini":
        
        # Handle async event loop issues with Gemini embeddings
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop (like Streamlit), use a thread
            def create_embedding_function():
                return GoogleGenerativeAIEmbeddings(
                    google_api_key=get_env_var("GEMINI_API_KEY"),
                    model=embedding_model
                )
            
            # Run in a separate thread to avoid event loop conflicts
            result = [None]
            exception = [None]
            
            def run_in_thread():
                try:
                    result[0] = create_embedding_function()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception[0]:
                raise exception[0]
            
            embedding_function = result[0]
            
        except RuntimeError:
            # No event loop running, safe to create directly
            embedding_function = GoogleGenerativeAIEmbeddings(
                google_api_key=get_env_var("GEMINI_API_KEY"),
                model=embedding_model
            )
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    
    logger.info(f"Loaded vector store from {persist_directory}")
    return vector_store

def generate_rag_answer(query, vector_store, n_results=5):
    """
    Generate an answer to a query using RAG
    """
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=n_results)
    
    # Extract feedback and metadata
    context_items = []
    for i, doc in enumerate(docs, 1):
        context_items.append(f"{i}. Feedback: {doc.page_content}")
        context_items.append(f"   NPS Score: {doc.metadata['nps_score']}")
        context_items.append(f"   Topic: {doc.metadata['topic']}")
        context_items.append(f"   Date: {doc.metadata['date']}")
        context_items.append("")
    
    context = "\n".join(context_items)
    
    # Create prompt for LLM
    prompt = f"""Based on the following customer feedback examples, please answer this question: "{query}"

Relevant customer feedback:
{context}

Please provide a comprehensive answer based only on the feedback provided above. Include specific examples from the feedback to support your points. If the feedback doesn't contain enough information to answer the question, please state that clearly.
"""
    
    # Generate answer using LLM
    api_provider = get_api_provider()
    try:
        llm_model = get_env_var("LLM_MODEL")
    except ValueError:
        llm_model = "gpt-3.5-turbo" if api_provider == "openai" else "gemini-1.5-flash"
    
    try:
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
                    {"role": "system", "content": "You are a helpful assistant that analyzes customer feedback and provides insights based on the provided examples."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                **extra_kwargs
            )
            answer = response.choices[0].message.content.strip()
        elif api_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=get_env_var("GEMINI_API_KEY"))
            model = genai.GenerativeModel(llm_model)
            full_prompt = f"You are a helpful assistant that analyzes customer feedback and provides insights based on the provided examples.\n\n{prompt}"
            response = model.generate_content(full_prompt)
            answer = response.text.strip()
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")
        
        # Create result object
        result = {
            'query': query,
            'answer': answer,
            'sources': [{
                'feedback_id': doc.metadata['feedback_id'],
                'feedback_text': doc.page_content,
                'nps_score': doc.metadata['nps_score'],
                'topic': doc.metadata['topic']
            } for doc in docs]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating RAG answer: {e}")
        return None

def generate_topic_insights(df, vector_store):
    """
    Generate insights for each topic using RAG
    """
    # Get unique topics, excluding noise
    topics = df[df['cluster'] != -1]['topic'].unique()
    
    insights = {}
    for topic in topics:
        # Create queries for this topic
        queries = [
            f"What are the main issues or pain points mentioned in the '{topic}' feedback?",
            f"What positive aspects are mentioned in the '{topic}' feedback?",
            f"What actionable recommendations can be made based on the '{topic}' feedback?"
        ]
        
        # Generate answers for each query
        topic_insights = {}
        for query in queries:
            result = generate_rag_answer(query, vector_store)
            if result:
                # Extract the key from the query
                if 'issues' in query or 'pain points' in query:
                    key = 'issues'
                elif 'positive' in query:
                    key = 'positives'
                elif 'recommendations' in query or 'actionable' in query:
                    key = 'recommendations'
                else:
                    key = 'general'
                
                topic_insights[key] = result['answer']
        
        insights[topic] = topic_insights
    
    return insights

def create_rag_system(input_file, output_dir=None, generate_insights=True):
    """
    Create a RAG system from labeled feedback data and generate insights
    """
    # Load data with topic labels
    df = load_feedback_data(input_file)
    if df is None:
        return None
    
    # Check if topic labels are present
    if 'topic' not in df.columns:
        logger.error(f"No topic labels found in {input_file}")
        return None
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        persist_directory = os.path.join(output_dir, 'vector_store')
    else:
        output_dir = os.path.dirname(input_file)
        persist_directory = None
    
    # Create vector store
    vector_store = create_vector_store(df, persist_directory)
    
    # Generate insights for each topic if requested
    if generate_insights:
        logger.info("Generating insights for each topic")
        insights = generate_topic_insights(df, vector_store)
        
        # Save insights to file
        insights_path = os.path.join(output_dir, 'topic_insights.json')
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2)
        
        logger.info(f"Saved topic insights to {insights_path}")
    
    return vector_store

def main():
    parser = argparse.ArgumentParser(description='Create RAG system from labeled feedback data')
    parser.add_argument('--input', required=True, help='Input CSV file with labeled feedback')
    parser.add_argument('--output-dir', help='Output directory for vector store and insights')
    parser.add_argument('--no-insights', action='store_true', help='Skip generating topic insights')
    parser.add_argument('--query', help='Query to answer using the RAG system')
    args = parser.parse_args()
    
    # If output directory is not specified, use a default based on the input file
    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.input), 'rag_output')
    
    # Create RAG system
    vector_store = create_rag_system(
        args.input, 
        args.output_dir, 
        not args.no_insights
    )
    
    if vector_store is not None:
        # Print summary
        print(f"\nRAG System Summary:")
        print(f"Vector store created with {vector_store._collection.count()} documents")
        print(f"Output directory: {args.output_dir}")
        
        # Answer query if specified
        if args.query:
            print(f"\nQuery: {args.query}")
            result = generate_rag_answer(args.query, vector_store)
            if result:
                print(f"\nAnswer: {result['answer']}")
                print(f"\nSources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['feedback_text']} (NPS: {source['nps_score']}, Topic: {source['topic']})")

if __name__ == "__main__":
    main()