import time
from datetime import datetime

def evaluate_retrieval(vector_store, test_queries, top_k=5):
    """
    Evaluate retrieval performance across multiple queries
    
    Args:
        vector_store: The vector store to test
        test_queries: List of test queries
        top_k: Number of results to retrieve
        
    Returns:
        Dictionary containing evaluation metrics
    """
    results = []
    total_time = 0
    
    for query in test_queries:
        start_time = time.time()
        
        # Perform search
        search_results = vector_store.similarity_search_with_score(query, k=top_k)
        
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        
        # Extract scores
        scores = [score for _, score in search_results]
        
        results.append({
            "query": query,
            "num_results": len(search_results),
            "query_time": query_time,
            "scores": scores,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "best_score": min(scores) if scores else 0,  # Lower is better for distance
            "results": search_results
        })
    
    avg_query_time = total_time / len(test_queries) if test_queries else 0
    
    evaluation_summary = {
        "total_queries": len(test_queries),
        "total_time": total_time,
        "avg_query_time": avg_query_time,
        "results": results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return evaluation_summary


def generate_evaluation_report(evaluations, model_name, db_type):
    """
    Generate a comprehensive evaluation report
    
    Args:
        evaluations: List of evaluation summaries
        model_name: Name of the embedding model used
        db_type: Type of vector database used
        
    Returns:
        Formatted report string
    """
    report = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║          RETRIEVAL EVALUATION REPORT                         ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Configuration:
    • Embedding Model: {model_name}
    • Vector Database: {db_type}
    • Evaluation Time: {evaluations['timestamp']}
    
    Performance Metrics:
    • Total Queries Tested: {evaluations['total_queries']}
    • Total Processing Time: {evaluations['total_time']:.4f} seconds
    • Average Query Time: {evaluations['avg_query_time']:.4f} seconds
    
    ══════════════════════════════════════════════════════════════
    Query-Level Analysis:
    ══════════════════════════════════════════════════════════════
    """
    
    for idx, result in enumerate(evaluations['results'], 1):
        report += f"""
    Query #{idx}: "{result['query']}"
    • Results Retrieved: {result['num_results']}
    • Query Time: {result['query_time']:.4f} seconds
    • Average Similarity Score: {result['avg_score']:.4f}
    • Best Match Score: {result['best_score']:.4f}
    """
    
    return report


def compare_models(evaluation_results):
    """
    Compare performance across different models/configurations
    
    Args:
        evaluation_results: Dictionary of {config_name: evaluation_summary}
        
    Returns:
        Comparison report
    """
    comparison = {
        "configurations": [],
        "avg_times": [],
        "avg_scores": []
    }
    
    for config_name, eval_data in evaluation_results.items():
        comparison["configurations"].append(config_name)
        comparison["avg_times"].append(eval_data["avg_query_time"])
        
        # Calculate average score across all queries
        all_scores = []
        for result in eval_data["results"]:
            all_scores.extend(result["scores"])
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        comparison["avg_scores"].append(avg_score)
    
    return comparison
