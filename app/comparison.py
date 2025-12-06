import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from gensim import downloader
from datetime import datetime
import os

# https://claude.ai/chat/33c0261b-9401-44df-8cf4-8d1f615968eb This chat helped me with loading word2vec models in gensim

# Load general word embeddings
print("Loading general model (GloVe)...")
general_model = downloader.load('glove-wiki-gigaword-200')

# Load specialized models
print("Loading Law2Vec model...")
law_path = 'Law2Vec.200d.txt'
law_model = KeyedVectors.load_word2vec_format(
    law_path,
    binary=False
)
so_path = 'SO_vectors_200.bin'
print("Loading StackOverflow model...")
so_model = KeyedVectors.load_word2vec_format(
    so_path,
    binary=True
)

# NOTE: To add Code2Vec or other code embeddings:
# You'll need a pre-trained word2vec format model trained on code
# Options:
# 1. Download from: https://github.com/Microsoft/CodeSearchNet or similar
# 2. Train your own using gensim on a code corpus
# 3. Use existing code embeddings in word2vec format
# 
# Example when you have the file:
# print("Loading Code embedding model...")
# code_model = KeyedVectors.load_word2vec_format(
#     r'c:\Repositories\word2vec_corpus_analysis\code_embeddings_200d.txt',
#     binary=False
# )


# https://gemini.google.com/app/0e73907524edf76a  This chat gave me the idea to implement Nearest Neighbor Overlap rather than Cosine Similarity
def get_word_level_nne_scores(model_a, model_b, top_n=10, sample_size=1000):
    """
    Calculate NNE for individual words between two models.
    Returns a list of NNE scores and corresponding words.

    Parameters:
    model_a (gensim.models.KeyedVectors): The first word embedding model.
    model_b (gensim.models.KeyedVectors): The second word embedding model.
    top_n (int): The number of nearest neighbors to consider for overlap calculation.
    sample_size (int): Number of common words to sample for comparison.

    Returns:
    tuple: (list of NNE scores, list of corresponding words)
    """
    common_words = list(set(model_a.index_to_key).intersection(set(model_b.index_to_key)))
    
    if not common_words:
        return [], []
    
    # Sample words if there are too many
    if len(common_words) > sample_size:
        common_words = np.random.choice(common_words, sample_size, replace=False)
    
    word_nne_scores = []
    valid_words = []
    
    print(f"Calculating NNE for {len(common_words)} words...")
    for i, word in enumerate(common_words):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(common_words)} words")
        
        try:
            neighbors_a = set([neighbor for neighbor, _ in model_a.most_similar(word, topn=top_n)])
            neighbors_b = set([neighbor for neighbor, _ in model_b.most_similar(word, topn=top_n)])
            
            overlap = len(neighbors_a.intersection(neighbors_b))
            nne_score = overlap / top_n  # NNE for this specific word
            word_nne_scores.append(nne_score)
            valid_words.append(word)
        except Exception as e:
            # Print first few errors to diagnose issues
            if len(word_nne_scores) < 5:
                print(f"  Error with word '{word}': {e}")
            continue
    
    return word_nne_scores, valid_words


def get_cosine_similarity_scores(model_a, model_b, sample_size=1000):
    """
    Calculate cosine similarity for individual words between two models.
    Returns a list of cosine similarity scores and corresponding words.

    Parameters:
    model_a (gensim.models.KeyedVectors): The first word embedding model.
    model_b (gensim.models.KeyedVectors): The second word embedding model.
    sample_size (int): Number of common words to sample for comparison.

    Returns:
    tuple: (list of cosine similarity scores, list of corresponding words)
    """
    common_words = list(set(model_a.index_to_key).intersection(set(model_b.index_to_key)))
    
    if not common_words:
        return [], []
    
    # Sample words if there are too many
    if len(common_words) > sample_size:
        common_words = np.random.choice(common_words, sample_size, replace=False)
    
    cosine_scores = []
    valid_words = []
    
    print(f"Calculating Cosine Similarity for {len(common_words)} words...")
    for i, word in enumerate(common_words):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(common_words)} words")
        
        try:
            # Get word vectors from both models
            vec_a = model_a[word]
            vec_b = model_b[word]
            
            # Calculate cosine similarity
            cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
            cosine_scores.append(cosine_sim)
            valid_words.append(word)
        except Exception as e:
            # Print first few errors to diagnose issues
            if len(cosine_scores) < 5:
                print(f"  Error with word '{word}': {e}")
            continue
    
    return cosine_scores, valid_words


def plot_nne_histogram(scores, title, filename=None, score_label='NNE Score'):
    """
    Plot histogram of scores.
    
    Parameters:
    scores (list): List of scores to plot.
    title (str): Title for the plot.
    filename (str): Optional filename to save the plot.
    score_label (str): Label for the x-axis (default: 'NNE Score').
    """
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, color='#2b2b2b', edgecolor='white', alpha=0.7)
    
    plt.xlabel(score_label, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
    plt.axvline(median_score, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_score:.3f}')
    plt.legend()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_top_bottom_words(scores, words, title, filename=None):
    """
    Plot bar chart showing top 10 and bottom 10 words by NNE score.
    
    Parameters:
    scores (list): List of NNE scores.
    words (list): List of corresponding words.
    title (str): Title for the plot.
    filename (str): Optional filename to save the plot.
    """
    # Combine scores and words, then sort
    word_score_pairs = list(zip(words, scores))
    word_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 10 and bottom 10
    top_10 = word_score_pairs[:10]
    bottom_10 = word_score_pairs[-10:]
    bottom_10.reverse()  # Reverse so lowest is at top
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot top 10
    top_words = [w for w, s in top_10]
    top_scores = [s for w, s in top_10]
    ax1.barh(range(len(top_words)), top_scores, color='#2ecc71', edgecolor='black')
    ax1.set_yticks(range(len(top_words)))
    ax1.set_yticklabels(top_words, fontsize=10)
    ax1.set_xlabel('NNE Score', fontsize=12)
    ax1.set_title('Top 10 Words (Highest Overlap)', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add score labels on bars
    for i, score in enumerate(top_scores):
        ax1.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=9)
    
    # Plot bottom 10
    bottom_words = [w for w, s in bottom_10]
    bottom_scores = [s for w, s in bottom_10]
    ax2.barh(range(len(bottom_words)), bottom_scores, color='#e74c3c', edgecolor='black')
    ax2.set_yticks(range(len(bottom_words)))
    ax2.set_yticklabels(bottom_words, fontsize=10)
    ax2.set_xlabel('NNE Score', fontsize=12)
    ax2.set_title('Bottom 10 Words (Lowest Overlap)', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add score labels on bars
    for i, score in enumerate(bottom_scores):
        ax2.text(score + 0.001, i, f'{score:.3f}', va='center', fontsize=9)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    # Create results folder with date
    results_dir = r'c:\Repositories\word2vec_corpus_analysis\results\finalV1'
    os.makedirs(results_dir, exist_ok=True)
    
    # Get current date for filenames
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Compare General vs Law2Vec
    print("\n=== Comparing General (GloVe) vs Law2Vec ===")
    
    # NNE Analysis
    law_scores, law_words = get_word_level_nne_scores(general_model, law_model, top_n=10, sample_size=1000)
    print(f"Average NNE Score: {np.mean(law_scores):.4f}")
    print(f"Median NNE Score: {np.median(law_scores):.4f}")
    
    # Plot NNE histogram
    plot_nne_histogram(law_scores, 'General vs Law2Vec - NNE Score Distribution', 
                       os.path.join(results_dir, f'law_nne_{date_str}.png'))
    
    # Plot top and bottom words for NNE
    plot_top_bottom_words(law_scores, law_words, 'General vs Law2Vec - Top & Bottom Words by NNE',
                         os.path.join(results_dir, f'law_nne_top_bottom_{date_str}.png'))
    
    # Cosine Similarity Analysis
    law_cosine_scores, law_cosine_words = get_cosine_similarity_scores(general_model, law_model, sample_size=1000)
    print(f"Average Cosine Similarity: {np.mean(law_cosine_scores):.4f}")
    print(f"Median Cosine Similarity: {np.median(law_cosine_scores):.4f}")
    
    # Plot cosine similarity histogram
    plot_nne_histogram(law_cosine_scores, 'General vs Law2Vec - Cosine Similarity Distribution', 
                       os.path.join(results_dir, f'law_cosine_{date_str}.png'), score_label='Cosine Similarity')
    
    # Plot top and bottom words for cosine similarity
    plot_top_bottom_words(law_cosine_scores, law_cosine_words, 'General vs Law2Vec - Top & Bottom Words by Cosine Similarity',
                         os.path.join(results_dir, f'law_cosine_top_bottom_{date_str}.png'))
    
    # Compare General vs Stack Overflow (Programming)
    print("\n=== Comparing General (GloVe) vs Stack Overflow (Programming) ===")
    
    # NNE Analysis
    so_scores, so_words = get_word_level_nne_scores(general_model, so_model, top_n=10, sample_size=1000)
    print(f"Average NNE Score: {np.mean(so_scores):.4f}")
    print(f"Median NNE Score: {np.median(so_scores):.4f}")
    
    # Plot NNE histogram
    plot_nne_histogram(so_scores, 'General vs Stack Overflow - NNE Score Distribution', 
                       os.path.join(results_dir, f'so_nne_{date_str}.png'))
    
    # Plot top and bottom words for NNE
    plot_top_bottom_words(so_scores, so_words, 'General vs Stack Overflow - Top & Bottom Words by NNE',
                         os.path.join(results_dir, f'so_nne_top_bottom_{date_str}.png'))
    
    # Cosine Similarity Analysis
    so_cosine_scores, so_cosine_words = get_cosine_similarity_scores(general_model, so_model, sample_size=1000)
    print(f"Average Cosine Similarity: {np.mean(so_cosine_scores):.4f}")
    print(f"Median Cosine Similarity: {np.median(so_cosine_scores):.4f}")
    
    # Plot cosine similarity histogram
    plot_nne_histogram(so_cosine_scores, 'General vs Stack Overflow - Cosine Similarity Distribution', 
                       os.path.join(results_dir, f'so_cosine_{date_str}.png'), score_label='Cosine Similarity')
    
    # Plot top and bottom words for cosine similarity
    plot_top_bottom_words(so_cosine_scores, so_cosine_words, 'General vs Stack Overflow - Top & Bottom Words by Cosine Similarity',
                         os.path.join(results_dir, f'so_cosine_top_bottom_{date_str}.png'))