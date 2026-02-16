# 🚀 Startup Failure Intelligence - RAG Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot powered by local LLM and vector search, analyzing 2,400+ startup failure records across multiple sectors.

## ✨ Features

- ✅ **RAG Architecture**: Combines vector search with local LLM for accurate context-based answers
- ✅ **2,400+ Records**: Comprehensive dataset of startup failures across 8 categories
- ✅ **Multi-Sector Analysis**: Finance, Healthcare, Manufacturing, Retail, Food, Information sectors
- ✅ **Local & Private**: Runs entirely on your machine - no API calls, no data sent to cloud
- ✅ **GPU Optimized**: Auto-detects CUDA for faster embeddings, falls back to CPU gracefully
- ✅ **Beautiful UI**: Streamlit interface with source citations and metadata
- ✅ **Zero Setup**: Automated vectorstore creation with one command
- ✅ **Fast Queries**: 2-5 second response times with context-aware answers

---

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed (for local LLM)

### Installation Steps

1. **Clone and Navigate**
   ```bash
   git clone https://github.com/AadithyaSJ/Rag-ChatBot.git
   cd RagChatBot
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download LLM Model** (in another terminal)
   ```bash
   ollama pull llama3
   ```

5. **Index Data** (first run only, takes 5-15 mins depending on CPU/GPU)
   ```bash
   python ingest.py
   ```

6. **Start Ollama Server** (new terminal, keep running)
   ```bash
   ollama serve
   ```

7. **Launch App**
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`

---

## 💬 Sample Questions

### Finance & Insurance Sector
- "Why did Theranos fail despite $700M funding?"
- "What are common financial issues that lead to startup failures?"
- "Why did Quibi shut down after $1.75B investment?"
- "What financial metrics correlate with startup success?"

### Healthcare Sector
- "Why did Babylon Health struggle?"
- "What challenges do healthtech startups face?"
- "Why did Proteus Digital Health pivot?"
- "What healthcare business models fail most commonly?"

### Manufacturing Sector
- "Why couldn't Fisker compete with Tesla?"
- "What manufacturing costs sink startups?"
- "Why did Jibo fail in robotics?"
- "How do supply chain issues affect manufacturers?"

### Retail & E-commerce
- "Why did Wish fade despite $1.8B funding?"
- "What causes retail startup failures?"
- "Why did Jet.com sell to Walmart?"
- "How do logistics costs impact retail startups?"

### Food & Services
- "Why did Sprig shut down?"
- "What food delivery models are sustainable?"
- "Why did Juicero fail?"
- "What operational challenges affect food startups?"

### Information & Tech Sector
- "Why did Twitter/X acquire and shut down startups?"
- "What core tech metrics predict failure?"
- "Why did Munchery pivot multiple times?"
- "How important is product-market fit?"

### Cross-Sector Analysis
- "What's the most common reason startups fail?"
- "How does funding relate to failure rates?"
- "Which sectors have highest failure rates?"
- "What's the average time to failure?"
- "What patterns exist across successful pivots?"
- "How important is team experience?"
- "What role does market timing play?"

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA INGESTION                          │
├─────────────────────────────────────────────────────────────┤
│  CSV Files (8 datasets, 2,442 rows)                        │
│          ↓                                                   │
│  Row → Text Conversion (metadata extraction)               │
│          ↓                                                   │
│  Sentence Transformers Embeddings (all-MiniLM-L6-v2)      │
│          ↓                                                   │
│  FAISS Vector Store (CPU/GPU optimized)                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    RAG QUERY PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│  User Question                                             │
│          ↓                                                   │
│  Vector Search (retrieve k=5 most similar documents)       │
│          ↓                                                   │
│  Ollama LLM (llama3, temperature=0.1)                      │
│          ↓                                                   │
│  Context-Aware Response + Source Citations                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                             │
├─────────────────────────────────────────────────────────────┤
│  • Query Input                                             │
│  • Answer Display                                          │
│  • Expandable Source Viewer (with metadata)               │
│  • Real-time Response Streaming                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Overview

| Dataset | Records | Sector | Status |
|---------|---------|--------|--------|
| Finance and Insurance | 47 | Finance | ✅ |
| Food and Services | 26 | Food | ✅ |
| Health Care | 60 | Healthcare | ✅ |
| Manufactures | 30 | Manufacturing | ✅ |
| Retail Trade | 90 | Retail | ✅ |
| Information Sector | 156 | Information & Tech | ✅ |
| General Failures | 815 | General | ✅ |
| Combined Dataset | 1,218 | Combined | ✅ |
| **TOTAL** | **2,442** | **All** | **✅** |

---

## 🎯 How It Works (3-Phase Process)

### Phase 1: Data Ingestion
1. Scan `data/` folder for all CSV files
2. Convert each row to structured text format
3. Extract metadata (dataset name, sector, years, etc.)
4. Generate embeddings using sentence-transformers
5. Store in FAISS vector database

**Performance**: ~2,442 documents indexed in 5-15 minutes (CPU), ~1-2 minutes (GPU)

### Phase 2: RAG Query
1. User asks a question in the Streamlit UI
2. Question converted to embedding
3. Vector search retrieves top 5 most relevant documents
4. Documents + question sent to Ollama LLM
5. LLM generates answer grounded in provided context

**Performance**: 2-5 seconds per query (after LLM warmup)

### Phase 3: Display
1. Answer rendered in markdown format
2. Source documents displayed as expandable cards
3. Each source shows: file path, company name, sector, preview text
4. User can review sources to verify claim accuracy

---

## ⚙️ Configuration

### Adjust Retrieval Count
Edit `rag_chain.py`:
```python
search_kwargs = {"k": 5}  # Change 5 to desired number of results
```

### Change LLM Temperature
Edit `rag_chain.py`:
```python
llm = Ollama(
    model="llama3",
    temperature=0.1,  # 0.0 = deterministic, 1.0 = creative
)
```

### Use Different Ollama Model
```bash
ollama pull mistral    # or neural-chat, openchat, etc.
```
Then edit `rag_chain.py`:
```python
llm = Ollama(model="mistral")
```

### Change Embedding Model
Edit `ingest.py` and `rag_chain.py`:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Larger, slower
)
```

---

## ⚡ Performance Tips

1. **First Query Warmup**: LLM initialization takes 5-10 seconds on first query. Subsequent queries are 2-5 seconds.

2. **CPU vs GPU**:
   - CPU: Good for testing, ~1-2 seconds per query
   - GPU (CUDA): 2-3x faster, auto-detected if available

3. **Smaller Models**: If Ollama is slow:
   ```bash
   ollama pull mistral      # Faster, less accurate
   ollama pull neural-chat  # Good balance
   ```

4. **Batch Queries**: Multiple questions benefit from LLM warmness - each subsequent query faster

5. **Monitor Memory**: Check resource usage in task manager/system monitor

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to Ollama"
**Solution**: Ensure Ollama server is running in separate terminal
```bash
ollama serve
```
Check if running on correct port (default 11434). Verify with:
```bash
curl http://localhost:11434/api/tags
```

### Issue: "FAISS vectorstore not found"
**Solution**: First-time setup requires ingestion
```bash
python ingest.py
```
This creates `vectorstore/faiss_index/` directory

### Issue: "Out of Memory (OOM) error"
**Solution 1**: Use smaller model
```bash
ollama pull orca-mini  # 3.3B params, ~2GB RAM
```

**Solution 2**: Reduce k parameter in queries
```python
search_kwargs = {"k": 3}  # Instead of 5
```

### Issue: "Ingestion is very slow"
**Solution 1**: Check if GPU is available and pytorch installed
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution 2**: For initial setup, CPU ingestion is normal (5-15 mins). Subsequent runs use cached index.

### Issue: "Poor answer quality"
**Solutions**:
- Ask more specific questions with context
- Retrieve more documents: change `k` from 5 to 7-10
- Ensure Ollama has warmed up (run 2-3 queries first)
- Try fact-based questions vs opinion questions

---

## 📁 Project Structure

```
RagChatBot/
├── app.py                          # Streamlit UI
├── ingest.py                       # Data ingestion pipeline
├── rag_chain.py                    # RAG chain implementation
├── ipl_agent.py                    # Agent wrapper
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                           # Input CSV files
│   ├── all_season_batting_card.csv
│   ├── all_season_bowling_card.csv
│   ├── all_season_details.csv
│   ├── all_season_summary.csv
│   ├── points_table.csv
│   ├── 2022/
│   ├── 2023/
│   └── 2024/
└── vectorstore/                    # FAISS index (generated)
    └── faiss_index/
```

---

## 🔒 Privacy & Security

✅ **100% Local Execution**: No data sent to external APIs
- LLM runs locally via Ollama
- Embeddings computed locally via sentence-transformers
- Vector search performed locally via FAISS

✅ **No Cloud Dependencies**: Except Ollama/HuggingFace model downloads

✅ **No API Keys Required**: Unlike OpenAI/Anthropic alternatives

⚠️ **Note**: First model download requires internet (llama3 ~4GB)

---

## 📚 External Resources

- **Ollama Documentation**: https://github.com/jmorganca/ollama
- **LangChain Docs**: https://docs.langchain.com
- **Streamlit Docs**: https://docs.streamlit.io
- **FAISS Guide**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net

---

## 🚀 Next Steps

1. Open the README and run the setup commands
2. Try sample questions from the "Sample Questions" section
3. Review `app.py` to understand the UI flow
4. Explore different Ollama models for different use cases
5. Fine-tune hyperparameters for your needs

---

## 📝 License

MIT License - Feel free to use and modify

---

## 💡 Tips for Best Results

1. **Be Specific**: "Why did Theranos fail?" → Better than "Tell me about startup failures"
2. **Use Context**: "Based on the data, what factors correlate with healthcare startup failures?"
3. **Follow-up Questions**: RAG maintains context for follow-ups
4. **Review Sources**: Always check the retrieved sources to validate answers

---

**Created by**: Aadithya SJ  
**Last Updated**: February 2026  
**Status**: ✅ Production Ready
