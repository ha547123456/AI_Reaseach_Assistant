# AI Research Assistant - Semantic Search Module


This project implements a complete semantic search engine with GUI for AI-powered document retrieval.

## Features

### GUI-Based Data Selection
- Upload multiple text documents (minimum 10)
- View dataset statistics (number of documents, total size, word counts)
- Preview document contents
- No hardcoded datasets - fully dynamic

### Embedding & Vector Store Configuration
- Select from multiple HuggingFace embedding models:
  - sentence-transformers/all-MiniLM-L6-v2 (fast, efficient)
  - sentence-transformers/all-mpnet-base-v2 (higher accuracy)
  - sentence-transformers/paraphrase-MiniLM-L6-v2 (paraphrase detection)
- Choose vector database (FAISS or Chroma)
- Automatic embedding generation and storage

### Semantic Retrieval
- Natural language query input
- Adjustable top-k results (1-10)
- Results ordered by semantic similarity
- Visual relevance scores
- Document metadata display
- Search history tracking

### Retrieval Evaluation & Analysis
- Multi-query batch evaluation
- Performance metrics:
  - Query processing time
  - Average similarity scores
  - Score distribution analysis
- Interactive dashboards with visualizations
- Model comparison capabilities
- Downloadable evaluation reports

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download the models (automatic on first use):**
The embedding models will be downloaded automatically when you first run the application.

## Project Structure

```
ai-semantic-search/
│
├── 📄 README.md                    # Main documentation
├── 📄 requirements.txt             # Python dependencies
│
├── 📂 src/                         # Source code
│   ├── app.py                      # Main Streamlit application
│   ├── data_loader.py              # Document processing
│   ├── embedding_manager.py        # Embeddings & vector store
│   └── evaluation.py               # Performance evaluation
│
├── 📂 docs/                   
│   └── TESTING_GUIDE.txt     
│
├── 📂 sample_data/              
│   ├── doc1_intro_to_ai.txt
│   ├── doc2_machine_learning.txt
│   ├── doc3_deep_learning.txt
│   └── ... (15 total)
│
├── 📂 scripts/                     
   └── create_sample_docs.py

```

## Usage

### 1. Start the Application

```bash
streamlit run app.py
```

Or:
```bash
python -m streamlit run app.py
```

### 2. Upload Documents 

1. Click "Browse files" or drag-and-drop at least 10 .txt files
2. View dataset statistics and document previews
3. Verify all documents loaded correctly

### 3. Configure Vector Store 

1. Select an embedding model from the dropdown
2. Choose a vector database (FAISS or Chroma)
3. Click " Generate Embeddings & Create Vector Store"
4. Wait for processing to complete

### 4. Perform Semantic Search 

1. Enter your natural language query
2. Adjust the "Top K Results" slider
3. Click " Search"
4. Review results ordered by relevance
5. Check similarity scores and document content

### 5. Run Evaluation 

#### Option A: Quick Evaluation
1. Navigate to " Run Evaluation" tab
2. Use predefined test queries or enter your own
3. Click " Run Evaluation"
4. Download the evaluation report

#### Option B: Analysis Dashboard
1. Go to " Analysis Dashboard" tab
2. View query performance metrics
3. Analyze score distributions
4. Review detailed results table

#### Option C: Model Comparison
1. Open " Model Comparison" tab
2. View search history
3. Compare different configurations
4. Read observations and recommendations

## Embedding Models Comparison

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| all-MiniLM-L6-v2 |   Fast | Good | General purpose, large datasets |
| all-mpnet-base-v2 |   Slower |   Best | High accuracy requirements |
| paraphrase-MiniLM-L6-v2 |   Fast | Good | Paraphrase detection |

## Vector Database Comparison

| Database | Speed | Persistence | Best For |
|----------|-------|-------------|----------|
| FAISS |   Faster | Memory-based | Large-scale similarity search |
| Chroma |   Moderate | Disk-based | Persistent storage, metadata filtering |

## Evaluation Metrics Explained

- **Query Time**: Time taken to process each query (lower is better)
- **Similarity Score**: Distance metric (lower means more similar)
- **Relevance %**: Converted similarity score (higher means more relevant)
- **Average Score**: Mean similarity across all retrieved documents
- **Best Score**: Highest similarity score in results

## Sample Test Queries

For academic/research documents:
- "What is artificial intelligence?"
- "Explain machine learning algorithms"
- "Deep learning neural networks"
- "Applications of natural language processing"
- "Computer vision techniques"

For general documents:
- Adapt queries based on your document content
- Use specific terms from your domain
- Test both broad and specific queries

## Troubleshooting

### Import Errors
If you get import errors for langchain modules:
```bash
pip install langchain-community langchain-text-splitters --upgrade
```

### FAISS Installation Issues
On Windows, if FAISS installation fails:
```bash
pip install faiss-cpu --no-cache-dir
```

### Memory Issues
For large datasets or models:
- Use MiniLM models instead of mpnet
- Reduce chunk_size in embedding_manager.py
- Process documents in smaller batches

### Slow Performance
- Use FAISS instead of Chroma
- Select all-MiniLM-L6-v2 model
- Reduce top_k value
- Use smaller document chunks

## Performance Tips

1. **For Speed**: Use FAISS + MiniLM-L6-v2
2. **For Accuracy**: Use Chroma + mpnet-base-v2
3. **For Balance**: Use FAISS + mpnet-base-v2

## Output Files

- **Evaluation Reports**: Text files with detailed metrics
- **Search History**: Stored in session state for comparison
- **Vector Stores**: Temporary (recreated each session)

## Future Enhancements

- Persistent vector store saving/loading
- Support for PDF, DOCX, and other file formats
- Advanced filtering and metadata search
- Hybrid search (semantic + keyword)
- Custom embedding model fine-tuning
- Multi-modal search (text + images)

## Credits

Built with:
- Streamlit (UI framework)
- LangChain (Document processing and retrieval)
- HuggingFace Transformers (Embedding models)
- FAISS/Chroma (Vector databases)
- Plotly (Data visualization)

## License

This project is for educational purposes.
