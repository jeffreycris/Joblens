# JobLens 🔍

A RAG (Retrieval-Augmented Generation) app that answers natural language questions about job market trends using real job description data.

Built to demonstrate LLM fluency, vector search, and end-to-end ML pipeline skills.

---

## What It Does

Ask questions like:
- *"What skills show up most in data scientist roles?"*
- *"What do fintech companies typically require?"*
- *"Which roles mention Python and SQL together?"*

JobLens finds the most relevant job descriptions from its database, passes them to a local LLM, and returns a grounded answer — no hallucination, sourced from real data.

---

## Architecture

```
User Question
     │
     ▼
Embed question (sentence-transformers/all-MiniLM-L6-v2)
     │
     ▼
FAISS vector search → Top 5 most relevant JD chunks
     │
     ▼
Prompt: [chunks + question] → Llama 3.1 (via Ollama)
     │
     ▼
Grounded Answer + Sources
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama 3.1 8B via Ollama (local, free) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, free) |
| Vector store | FAISS |
| Framework | LangChain |
| UI | Streamlit |
| Data | 172 real job descriptions (Data Science / ML / Analytics roles) |

---

## Setup

**Requirements:** Python 3.9+, [Ollama](https://ollama.com) installed and running

```bash
# Clone the repo
git clone https://github.com/yourusername/joblens.git
cd joblens

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull the LLM
ollama pull llama3.1
```

**Add your data:** Place your job descriptions Excel file at the path defined in `load_data.py`.

```bash
# Build the vector index (run once)
python3 embed.py

# Launch the app
streamlit run app.py
```

---

## Key Concepts Demonstrated

- **RAG pipeline** — retrieval-augmented generation to ground LLM answers in real data
- **Chunking strategy** — 500-char chunks with 50-char overlap to preserve context at cut points
- **Vector similarity search** — cosine similarity over embeddings to find relevant content
- **Local LLM inference** — running Llama 3.1 fully on-device via Ollama (no API cost)
- **Hallucination reduction** — prompt design that restricts the LLM to retrieved context only
- **Retrieval evaluation** — source documents exposed in UI for answer traceability
