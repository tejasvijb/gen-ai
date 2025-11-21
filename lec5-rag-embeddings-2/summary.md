## 🎯 PRODUCTION RAG CHECKLIST

1. CHUNKING
   ✓ Use semantic boundaries (paragraphs, sections)
   ✓ Add 10-20% overlap between chunks
   ✓ Keep chunks 300-800 tokens
   ✓ Store chunk metadata (source, page, date)

2. EMBEDDINGS
   ✓ Use consistent models (don't mix)
   ✓ Batch embedding generation (cost-effective)
   ✓ Cache embeddings (don't regenerate)
   ✓ Version your embedding model

3. RETRIEVAL
   ✓ Retrieve 5-10 candidates, use top 3-5
   ✓ Use metadata filtering when possible
   ✓ Implement hybrid search (keyword + semantic)
   ✓ Rerank results for quality

4. GENERATION
   ✓ Include source citations
   ✓ Set appropriate temperature (0.3-0.7)
   ✓ Implement fallback responses
   ✓ Monitor token usage

5. EVALUATION
   ✓ Track retrieval accuracy
   ✓ Measure answer relevance
   ✓ Log failures for improvement
   ✓ A/B test changes


## 🔍 Retrieval & RAG Troubleshooting Guide

### **Poor retrieval (wrong docs)**
- → Improve chunking strategy  
- → Try query rephrasing  
- → Add metadata filters  
- → Use hybrid search  

### **Good retrieval, poor answers**
- → Optimize prompt engineering  
- → Adjust context window  
- → Rerank retrieved docs  
- → Use better LLM model  

### **Slow performance**
- → Cache embeddings  
- → Batch operations  
- → Use smaller embedding model  
- → Implement async processing  

### **High costs**
- → Use smaller models (text-embedding-3-small)  
- → Cache frequently asked questions  
- → Batch embed documents  
- → Implement rate limiting  
