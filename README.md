# word2vec_corpus_analysis
By comparing General Purpose Word2Vec against models trained on specialized domain corpora, this project illustrates the power of linear algebra in semantic modeling through Nearest Neighbor Overlap (NNE) and Cosine Similarity metrics.

### Environment
Language: Python 3.11.0
Package manager: pip
IDE: VSCode

### Libraries
- **Gensim** (4.4.0) - Word embedding model loading and similarity calculations
- **NumPy** (2.3.5) - Vector operations and mathematical computations
- **Matplotlib** (3.10.7) - Visualization and plotting

### Data Models
**General models:**
- glove-wiki-gigaword-200

**Specialized models:**
- Law2Vec (200d) - Legal domain embeddings
- SO_vectors_200 - Stack Overflow programming embeddings

### Metrics
- **NNE (Nearest Neighbor Overlap)** - Measures semantic neighborhood similarity between models
- **Cosine Similarity** - Measures vector angle similarity between word embeddings
    
