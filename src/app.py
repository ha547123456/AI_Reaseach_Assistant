import streamlit as st
from data_loader import process_documents
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title(" AI Research Assistant - Semantic Search Module")
st.markdown("### Phase 1: Dataset Upload & Inspection")

uploaded_files = st.file_uploader(
    "Upload at least 10 text documents (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:

    num_docs = len(uploaded_files)

    if num_docs < 10:
        st.error(" Please upload at least 10 documents.")
    else:
        st.success(" Dataset uploaded successfully!")

        documents, total_size, total_words, avg_words, doc_stats = process_documents(uploaded_files)

        # Store for future tasks (Task 2)
        st.session_state["documents"] = documents

        # Convert size to KB
        size_kb = total_size / 1024

        st.markdown("##  Dataset Information")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Number of Documents", num_docs)
        col2.metric("Total Size (KB)", f"{size_kb:.2f}")
        col3.metric("Total Words", total_words)
        col4.metric("Average Words per Doc", f"{avg_words:.2f}")

        st.markdown("##  Document Details")

        for doc in doc_stats:
            with st.expander(f" {doc['name']}"):
                st.write(f"Size: {doc['size']} characters")
                st.write(f"Words: {doc['words']}")
                
        st.markdown("##  Document Preview")

        for i, file in enumerate(uploaded_files):
            with st.expander(f"Preview: {file.name}"):
                content = documents[i]
                st.text(content[:500])  # show first 500 characters

from embedding_manager import create_vector_store, semantic_search
from evaluation import evaluate_retrieval, generate_evaluation_report, compare_models

if "documents" in st.session_state:

    st.markdown("---")
    st.markdown("##  Task 2: Embedding & Vector Store Configuration")

    col1, col2 = st.columns(2)
    
    with col1:
        # Embedding model selection
        model_name = st.selectbox(
            "Select HuggingFace Embedding Model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-MiniLM-L6-v2"
            ]
        )
    
    with col2:
        # Vector database selection
        db_type = st.radio(
            "Select Vector Database",
            ["FAISS", "Chroma"]
        )

    if st.button(" Generate Embeddings & Create Vector Store"):

        with st.spinner("Generating embeddings... This may take a minute."):

            vector_store = create_vector_store(
                st.session_state["documents"],
                model_name,
                db_type
            )

            st.session_state["vector_store"] = vector_store
            st.session_state["model_name"] = model_name
            st.session_state["db_type"] = db_type

        st.success(" Vector Store Created Successfully!")
        st.info(f" Model: {model_name} | Database: {db_type}")


# Task 3: Semantic Retrieval
if "vector_store" in st.session_state:
    
    st.markdown("---")
    st.markdown("##  Task 3: Semantic Retrieval")
    
    # Create two columns for query input and settings
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., What is machine learning?",
            help="Enter a natural language query to search semantically across documents"
        )
    
    with col2:
        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of most relevant documents to retrieve"
        )
    
    if st.button(" Search") and query:
        
        with st.spinner("Searching..."):
            results = semantic_search(
                st.session_state["vector_store"],
                query,
                top_k
            )
        
        st.success(f"Found {len(results)} results!")
        
        # Display results
        st.markdown("###  Search Results (Ordered by Relevance)")
        
        for idx, (doc, score) in enumerate(results, 1):
            with st.expander(f" Rank #{idx} | Similarity Score: {score:.4f}"):
                st.markdown(f"**Content:**")
                st.write(doc.page_content)
                st.markdown(f"**Metadata:**")
                st.json(doc.metadata)
                
                # Visual score indicator
                score_percentage = max(0, (1 - score) * 100)  # Convert distance to percentage
                st.progress(score_percentage / 100)
                st.caption(f"Relevance: {score_percentage:.2f}%")
        
        # Store search results for evaluation
        if "search_history" not in st.session_state:
            st.session_state["search_history"] = []
        
        st.session_state["search_history"].append({
            "query": query,
            "top_k": top_k,
            "results": results,
            "model": st.session_state["model_name"],
            "db": st.session_state["db_type"]
        })


# Task 4: Retrieval Evaluation and Analysis
if "vector_store" in st.session_state:
    
    st.markdown("---")
    st.markdown("##  Task 4: Retrieval Evaluation and Analysis")
    
    tab1, tab2, tab3 = st.tabs([" Run Evaluation", " Analysis Dashboard", " Model Comparison"])
    
    with tab1:
        st.markdown("### Test Multiple Queries")
        
        # Predefined test queries
        st.markdown("**Predefined Test Queries:**")
        default_queries = st.text_area(
            "Enter test queries (one per line):",
            value="""What is artificial intelligence?
How does machine learning work?
Explain neural networks
What are the applications of AI?
Deep learning techniques""",
            height=150
        )
        
        eval_top_k = st.slider(
            "Top K for Evaluation",
            min_value=1,
            max_value=10,
            value=5,
            key="eval_top_k"
        )
        
        if st.button(" Run Evaluation"):
            
            test_queries = [q.strip() for q in default_queries.split("\n") if q.strip()]
            
            if not test_queries:
                st.warning("Please enter at least one test query.")
            else:
                with st.spinner(f"Evaluating {len(test_queries)} queries..."):
                    evaluation_results = evaluate_retrieval(
                        st.session_state["vector_store"],
                        test_queries,
                        eval_top_k
                    )
                    
                    st.session_state["evaluation_results"] = evaluation_results
                
                st.success(f" Evaluation complete! Tested {len(test_queries)} queries.")
                
                # Generate and display report
                report = generate_evaluation_report(
                    evaluation_results,
                    st.session_state["model_name"],
                    st.session_state["db_type"]
                )
                
                st.text(report)
                
                # Download report
                st.download_button(
                    label=" Download Evaluation Report",
                    data=report,
                    file_name=f"evaluation_report_{st.session_state['model_name'].split('/')[-1]}_{st.session_state['db_type']}.txt",
                    mime="text/plain"
                )
    
    with tab2:
        st.markdown("### Performance Analysis Dashboard")
        
        if "evaluation_results" in st.session_state:
            eval_data = st.session_state["evaluation_results"]
            
            # Metrics overview
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Queries", eval_data["total_queries"])
            col2.metric("Avg Query Time", f"{eval_data['avg_query_time']:.4f}s")
            col3.metric("Total Time", f"{eval_data['total_time']:.4f}s")
            
            # Query performance visualization
            st.markdown("#### Query Performance")
            
            query_names = [f"Q{i+1}" for i in range(len(eval_data['results']))]
            query_times = [r['query_time'] for r in eval_data['results']]
            avg_scores = [r['avg_score'] for r in eval_data['results']]
            
            # Time chart
            fig_time = go.Figure()
            fig_time.add_trace(go.Bar(
                x=query_names,
                y=query_times,
                name='Query Time',
                marker_color='indianred'
            ))
            fig_time.update_layout(
                title="Query Processing Time",
                xaxis_title="Query",
                yaxis_title="Time (seconds)",
                height=400
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Score distribution
            st.markdown("#### Score Distribution")
            all_scores = []
            for result in eval_data['results']:
                all_scores.extend(result['scores'])
            
            fig_scores = px.histogram(
                all_scores,
                nbins=20,
                title="Similarity Score Distribution",
                labels={'value': 'Similarity Score', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_scores, use_container_width=True)
            
            # Detailed results table
            st.markdown("#### Detailed Results")
            results_df = pd.DataFrame([
                {
                    "Query": r["query"][:50] + "..." if len(r["query"]) > 50 else r["query"],
                    "Num Results": r["num_results"],
                    "Query Time (s)": f"{r['query_time']:.4f}",
                    "Avg Score": f"{r['avg_score']:.4f}",
                    "Best Score": f"{r['best_score']:.4f}"
                }
                for r in eval_data['results']
            ])
            st.dataframe(results_df, use_container_width=True)
            
        else:
            st.info(" Run an evaluation first to see the analysis dashboard.")
    
    with tab3:
        st.markdown("### Compare Different Configurations")
        
        st.markdown("""
        This section allows you to compare retrieval performance across:
        - Different embedding models
        - Different vector databases
        - Different datasets
        """)
        
        if "search_history" in st.session_state and len(st.session_state["search_history"]) > 0:
            st.markdown("#### Search History")
            
            history_df = pd.DataFrame([
                {
                    "Query": h["query"],
                    "Top K": h["top_k"],
                    "Model": h["model"].split('/')[-1],
                    "Database": h["db"],
                    "Num Results": len(h["results"])
                }
                for h in st.session_state["search_history"]
            ])
            
            st.dataframe(history_df, use_container_width=True)
        
        st.markdown("#### Observations & Recommendations")
        
        if "evaluation_results" in st.session_state:
            eval_data = st.session_state["evaluation_results"]
            
            st.markdown(f"""
            **Current Configuration:**
            - Model: `{st.session_state['model_name']}`
            - Database: `{st.session_state['db_type']}`
            - Average Query Time: `{eval_data['avg_query_time']:.4f}s`
            
            **Key Observations:**
            1. **Performance**: {'Fast' if eval_data['avg_query_time'] < 0.5 else 'Moderate' if eval_data['avg_query_time'] < 1.0 else 'Slow'} query processing
            2. **Consistency**: Score variance indicates retrieval stability
            3. **Scalability**: Consider FAISS for larger datasets (>1000 docs)
            
            **Recommendations:**
            - For speed: Use MiniLM models with FAISS
            - For accuracy: Use mpnet models with Chroma
            - For balance: Current configuration works well
            """)
        else:
            st.info("Run an evaluation to see detailed observations and recommendations.")


st.markdown("---")
st.markdown("###  Tips")
st.markdown("""
- Use specific queries for better results
- Higher top-k values give more context but may include less relevant results
- Test with multiple queries to evaluate consistency
- Compare different embedding models to find the best fit for your data
""")

