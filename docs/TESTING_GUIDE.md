# Testing Guide for Task 4: Retrieval Evaluation and Analysis

## Overview
This guide provides comprehensive testing scenarios to evaluate your semantic search system across different configurations.

## Pre-Testing Setup

### 1. Generate Sample Documents
```bash
python create_sample_docs.py
```
This creates 15 sample documents about AI/ML topics in the `sample_documents` directory.

### 2. Start the Application
```bash
streamlit run app.py
```

## Testing Scenarios

### Scenario 1: Single Model Evaluation (Basic)

**Objective**: Test basic retrieval functionality

**Steps**:
1. Upload all 15 sample documents
2. Select model: `sentence-transformers/all-MiniLM-L6-v2`
3. Select database: `FAISS`
4. Generate embeddings
5. Test with these queries:
   - "What is artificial intelligence?"
   - "Explain machine learning"
   - "How do neural networks work?"
   - "What are AI applications?"
   - "Ethical concerns in AI"

**Expected Results**:
- Query time < 1 second
- Relevant documents ranked first
- Similarity scores should vary (not all the same)

**Documentation**:
- Take screenshots of search results
- Note the top-3 documents for each query
- Record query processing times

---

### Scenario 2: Model Comparison

**Objective**: Compare different embedding models

**Test A - MiniLM Model**:
1. Configuration:
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Database: `FAISS`
   - Top-K: 5

2. Test queries:
   ```
   What is deep learning?
   Explain supervised learning
   Computer vision applications
   ```

3. Record:
   - Average query time
   - Average similarity scores
   - Top result for each query

**Test B - MPNet Model**:
1. Clear vector store (refresh page)
2. Upload same documents
3. Configuration:
   - Model: `sentence-transformers/all-mpnet-base-v2`
   - Database: `FAISS`
   - Top-K: 5

4. Test same queries as Test A

5. Record same metrics

**Comparison Analysis**:
Create a table:

| Metric | MiniLM | MPNet |
|--------|--------|-------|
| Avg Query Time | ? | ? |
| Avg Score | ? | ? |
| Most Relevant? | ? | ? |

---

### Scenario 3: Database Comparison

**Objective**: Compare FAISS vs Chroma

**Test A - FAISS**:
1. Configuration:
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Database: `FAISS`

2. Run batch evaluation with 10 queries
3. Note total processing time
4. Record memory usage (if possible)

**Test B - Chroma**:
1. Same model, different database
2. Run same 10 queries
3. Compare results

**Expected Differences**:
- FAISS: Faster, memory-based
- Chroma: Slower, persistent storage

---

### Scenario 4: Top-K Sensitivity Analysis

**Objective**: Understand how top-k affects results

**Steps**:
1. Use one configuration
2. Test query: "What are neural networks?"
3. Test with different k values:
   - k=1: Note the single result
   - k=3: Note top 3 results
   - k=5: Note top 5 results
   - k=10: Note top 10 results

**Analysis Questions**:
- At what k does relevance drop significantly?
- Are results 6-10 still relevant?
- What's the optimal k for your dataset?

---

### Scenario 5: Comprehensive Batch Evaluation

**Objective**: Run full evaluation suite

**Setup**:
1. Select your preferred configuration
2. Go to "Task 4: Run Evaluation" tab
3. Use these 15 test queries:

```
What is artificial intelligence?
How does machine learning work?
Explain neural networks
What are the applications of AI?
Deep learning techniques
Natural language processing overview
Computer vision technology
Reinforcement learning principles
Ethics in AI development
Supervised learning methods
Unsupervised learning algorithms
Data science fundamentals
Future of artificial intelligence
AI algorithm types
Challenges in AI research
```

**Execution**:
1. Run evaluation
2. Download the report
3. Analyze the dashboard

**Key Metrics to Document**:
- Total queries tested: 15
- Average query time: ?
- Fastest query: ?
- Slowest query: ?
- Score distribution (check histogram)
- Queries with best/worst retrieval

---

### Scenario 6: Edge Cases

**Objective**: Test system robustness

**Test Cases**:

1. **Vague Query**:
   - Query: "technology"
   - Expected: Should return multiple relevant docs

2. **Very Specific Query**:
   - Query: "convolutional neural networks for image processing"
   - Expected: Should return relevant doc even if exact phrase not present

3. **Misspelled Query**:
   - Query: "machne lerning"
   - Expected: May or may not work (test embedding robustness)

4. **Long Query**:
   - Query: "I want to understand how artificial intelligence systems use machine learning algorithms to process natural language and make predictions about future events"
   - Expected: Should extract key concepts

5. **Out-of-Domain Query**:
   - Query: "how to cook pasta"
   - Expected: Low scores, possibly no relevant results

---

## Evaluation Report Structure

### Section 1: Configuration Details
```
- Dataset: [Number of documents, total size]
- Embedding Model: [Model name]
- Vector Database: [FAISS/Chroma]
- Test Date: [Date and time]
```

### Section 2: Quantitative Results
```
- Total Queries: 
- Average Query Time: 
- Min Query Time: 
- Max Query Time: 
- Average Similarity Score: 
- Score Standard Deviation: 
```

### Section 3: Qualitative Analysis

**Strengths**:
- What worked well?
- Which types of queries performed best?
- Were results consistently relevant?

**Weaknesses**:
- What didn't work well?
- Which queries had poor results?
- Any unexpected behaviors?

### Section 4: Model Comparison

| Aspect | MiniLM-L6-v2 | MPNet-base-v2 | Paraphrase-MiniLM |
|--------|--------------|---------------|-------------------|
| Speed | | | |
| Accuracy | | | |
| Best For | | | |

### Section 5: Database Comparison

| Aspect | FAISS | Chroma |
|--------|-------|--------|
| Speed | | |
| Ease of Use | | |
| Best For | | |

### Section 6: Recommendations

Based on testing:
1. **For your dataset**: Which configuration works best?
2. **For speed**: What to use?
3. **For accuracy**: What to use?
4. **For production**: What would you recommend?

---

## Advanced Testing

### 1. Different Dataset Sizes

Test with:
- 10 documents (minimum)
- 20 documents
- 50 documents

Observe how performance scales.

### 2. Different Document Types

If available, test with:
- Academic papers
- News articles
- Technical documentation
- Creative writing

Note how different content types affect retrieval.

### 3. Cross-Domain Queries

If you have mixed content:
- Test queries from different domains
- Check if the system retrieves from the correct domain

---

## Checklist for Complete Evaluation

- [ ] Uploaded minimum 10 documents
- [ ] Tested at least 2 different embedding models
- [ ] Tested both FAISS and Chroma
- [ ] Ran batch evaluation with 10+ queries
- [ ] Tested different top-k values (1, 3, 5, 10)
- [ ] Documented query times
- [ ] Documented similarity scores
- [ ] Analyzed score distributions
- [ ] Tested edge cases
- [ ] Downloaded evaluation report
- [ ] Created comparison tables
- [ ] Wrote observations and recommendations
- [ ] Took screenshots for documentation

---

## Common Issues and Solutions

### Issue 1: All Similarity Scores Are Similar
**Cause**: Documents are too similar or queries are too generic
**Solution**: Use more diverse documents or more specific queries

### Issue 2: Slow Query Processing
**Cause**: Large model or many documents
**Solution**: Use MiniLM model or reduce chunk_size

### Issue 3: Irrelevant Results
**Cause**: Poor embedding quality or out-of-domain query
**Solution**: Try different model or refine query

### Issue 4: Low Similarity Scores
**Cause**: Query doesn't match document content
**Solution**: Verify query relevance to dataset

---

## Tips for Best Results

1. **Use Specific Queries**: "deep learning for image classification" vs "AI"
2. **Test Multiple Variations**: Try synonyms and rephrasing
3. **Document Everything**: Keep detailed notes for your report
4. **Compare Systematically**: Use same queries across different configs
5. **Analyze Patterns**: Look for trends in what works/doesn't work
6. **Be Critical**: Note both strengths and weaknesses
7. **Use Visualizations**: Dashboard charts help identify patterns

---

## Report Writing Tips

### Introduction
- Briefly explain the semantic search system
- State the objectives of evaluation
- Describe the dataset used

### Methodology
- Explain testing approach
- List configurations tested
- Describe evaluation metrics

### Results
- Present quantitative data (tables, charts)
- Show example queries and results
- Include screenshots

### Analysis
- Interpret the results
- Compare different configurations
- Discuss strengths and limitations

### Conclusion
- Summarize key findings
- Provide recommendations
- Suggest future improvements

### Appendix
- Full evaluation reports
- Complete query lists
- Additional screenshots

---

## Sample Queries by Category

### General AI Concepts
- "What is artificial intelligence?"
- "Define machine learning"
- "AI vs machine learning differences"

### Technical Details
- "How do neural networks learn?"
- "Backpropagation algorithm explanation"
- "Gradient descent optimization"

### Applications
- "AI in healthcare"
- "Computer vision use cases"
- "NLP applications"

### Ethical/Social
- "AI bias and fairness"
- "Privacy concerns in AI"
- "Job automation impact"

### Advanced Topics
- "Transfer learning techniques"
- "Generative adversarial networks"
- "Attention mechanisms in transformers"

---

Good luck with your evaluation! 
