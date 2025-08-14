import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import hdbscan
import logging
from utils import load_feedback_data, save_feedback_data

logger = logging.getLogger(__name__)

def parse_embedding(embedding_str):
    """
    Parse a string representation of an embedding back to a numpy array
    """
    try:
        # Remove brackets and split by commas
        if isinstance(embedding_str, str):
            # Remove brackets and split
            values = embedding_str.strip('[]').split()
            # Convert to float array
            return np.array([float(x) for x in values if x])
        elif isinstance(embedding_str, list):
            return np.array(embedding_str)
        else:
            return embedding_str
    except Exception as e:
        logger.error(f"Error parsing embedding: {e}")
        return np.array([])

def cluster_hdbscan(embeddings, min_cluster_size=2):
    """
    Cluster embeddings using HDBSCAN
    """
    # Convert embeddings to numpy array if it's not already
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings.tolist())
    
    # Check if embeddings is empty
    if len(embeddings) == 0:
        logger.error("No embeddings found for clustering")
        return np.array([])
    
    # Check if all embeddings are valid
    valid_embeddings = []
    valid_indices = []
    
    for i, emb in enumerate(embeddings):
        if isinstance(emb, np.ndarray) and len(emb) > 0 and not np.isnan(emb).any():
            valid_embeddings.append(emb)
            valid_indices.append(i)
    
    if len(valid_embeddings) == 0:
        logger.error("No valid embeddings found for clustering")
        return np.array([])
    
    valid_embeddings = np.array(valid_embeddings)
    
    logger.info(f"Clustering {len(valid_embeddings)} valid embeddings out of {len(embeddings)} total")
    
    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    cluster_labels = clusterer.fit_predict(valid_embeddings)
    
    # Create a full array of cluster labels, including -1 for invalid embeddings
    full_cluster_labels = np.full(len(embeddings), -1)
    for i, idx in enumerate(valid_indices):
        full_cluster_labels[idx] = cluster_labels[i]
    
    return full_cluster_labels

def visualize_clusters(df, output_dir):
    """
    Visualize clusters using t-SNE
    """
    # Filter out noise points
    df_filtered = df[df['cluster'] != -1].copy()
    
    if len(df_filtered) == 0:
        logger.warning("No valid clusters found for visualization")
        return
    
    # Get embeddings
    embeddings = np.array(df_filtered['embedding'].tolist())
    
    # Reduce dimensionality with t-SNE
    # Set perplexity to be less than n_samples
    perplexity = min(30, len(embeddings) - 1, 5)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create a DataFrame for visualization
    df_vis = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': df_filtered['cluster'].values,
        'topic': df_filtered['topic'].values if 'topic' in df_filtered.columns else 'Unknown'
    })
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot points colored by cluster
    sns.scatterplot(
        x='x', y='y', 
        hue='cluster', 
        data=df_vis,
        palette='viridis',
        s=100,
        alpha=0.7
    )
    
    # Add topic labels for cluster centers
    for cluster_id in df_vis['cluster'].unique():
        cluster_data = df_vis[df_vis['cluster'] == cluster_id]
        center_x = cluster_data['x'].mean()
        center_y = cluster_data['y'].mean()
        
        # Get the most common topic for this cluster
        if 'topic' in df_filtered.columns:
            topic = cluster_data['topic'].mode()[0]
            plt.text(
                center_x, center_y, 
                f"Cluster {cluster_id}: {topic}",
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7)
            )
    
    plt.title('Feedback Clusters Visualization')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend(title='Cluster ID')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'cluster_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Cluster visualization saved to {os.path.join(output_dir, 'cluster_visualization.png')}")

def cluster_feedback(input_file, output_file, min_cluster_size=2, visualize=True):
    """
    Cluster feedback based on embeddings
    """
    logger.info(f"Clustering with HDBSCAN, min_cluster_size={min_cluster_size}")
    
    # Load feedback data
    df = load_feedback_data(input_file)
    
    if df is None:
        logger.error(f"Failed to load feedback data from {input_file}")
        return None
    
    # Parse embeddings from string to numpy arrays
    df['embedding'] = df['embedding'].apply(parse_embedding)
    
    # Filter out rows with empty embeddings
    df = df[df['embedding'].apply(lambda x: len(x) > 0)]
    
    if len(df) == 0:
        logger.error("No valid embeddings found in the data")
        return None
    
    # Perform clustering
    cluster_labels = cluster_hdbscan(df['embedding'], min_cluster_size=min_cluster_size)
    
    # Add cluster labels to DataFrame
    df['cluster'] = cluster_labels
    
    # Log cluster statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    logger.info(f"Found {n_clusters} clusters and {n_noise} noise points")
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            continue
        count = list(cluster_labels).count(cluster_id)
        logger.info(f"Cluster {cluster_id}: {count} feedback entries")
    
    # Save clustered data
    save_feedback_data(df, output_file)
    logger.info(f"Clustered feedback saved to {output_file}")
    
    # Visualize clusters if requested
    if visualize:
        output_dir = os.path.dirname(output_file)
        visualize_clusters(df, output_dir)
    
    return df