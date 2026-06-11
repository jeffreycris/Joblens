"""
rag.py

The core RAG pipeline:
1. Load the saved FAISS index
2. Take a user question
3. Find the most relevant JD chunks (retrieval)
4. Send chunks + question to Llama via Ollama (generation)
5. Return the answer
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

INDEX_PATH = "joblens_index"
OLLAMA_MODEL = "llama3.1"

# How many chunks to retrieve per query
# More = more context for the LLM, but slower
TOP_K = 5


def load_qa_chain():
    """Load the FAISS index and set up the RAG chain. Call once at startup."""

    # Load the same embedding model used to build the index
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load the FAISS index from disk
    print("Loading FAISS index...")
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # safe — we built this index ourselves
    )

    # Set up retriever — finds top-k most similar chunks for any query
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Connect to Llama running locally via Ollama
    llm = Ollama(model=OLLAMA_MODEL)

    # The prompt template tells the LLM how to behave.
    # We pass in {context} (retrieved chunks) and {question} (user query).
    # Telling it to ONLY use the context prevents hallucination.
    prompt_template = """You are JobLens, an AI assistant that answers questions
about job market trends using real job descriptions.

Use ONLY the job description excerpts below to answer the question.
If the answer isn't in the excerpts, say "I don't have enough data to answer that."

IMPORTANT: These excerpts are only a small sample of a much larger dataset.
Never report counts, totals, percentages, or "most common" claims about the
whole job market — you cannot know them from this sample. If the question asks
how many, what share, or which is most common, say "I don't have enough data
to answer that" and describe only what the excerpts themselves show.

Job Description Excerpts:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # RetrievalQA ties it all together:
    # question -> retriever -> chunks -> prompt -> LLM -> answer
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",         # "stuff" = dump all chunks into one prompt
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain


def ask(chain, question):
    """Ask a question and return the answer + source chunks."""
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    return answer, sources


if __name__ == "__main__":
    chain = load_qa_chain()
    print("\nJobLens ready. Ask a question about your job market.\n")

    question = "What skills show up most often in data scientist roles?"
    print(f"Q: {question}")
    print("Thinking...")

    answer, sources = ask(chain, question)
    print(f"\nA: {answer}")
    print(f"\n({len(sources)} chunks retrieved)")
