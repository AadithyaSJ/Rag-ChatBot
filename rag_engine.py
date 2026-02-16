import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Chroma collection
chroma_client = chromadb.Client()
collection = chroma_client.get_collection(name="startup_failures")

# Load local LLM
llm = OllamaLLM(model="llama3")


def ask_question(query):
    # Embed query
    query_embedding = embedding_model.encode([query]).tolist()[0]

    # Retrieve top 3 similar documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
You are a startup failure analyst.

Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    response = llm.invoke(prompt)

    return response
