from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

VECTOR_PATH = "vectorstore/faiss_index"


def _get_embedding_device():
    """Auto-detect GPU if available."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_rag_chain():
    """Create a RAG chain for startup failure analysis."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": _get_embedding_device()},
    )
    
    vectorstore = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = OllamaLLM(model="llama3", temperature=0.1)

    prompt_template = """You are an AI assistant analyzing startup failure data.

Use the context provided below to answer questions about startup failures, sectors, funding, and reasons for failure.

If the answer cannot be found in the context, say: "I don't have that information in the startup failure dataset."

Be concise and factual. When relevant, mention the startup name, sector, funding amount, and reason for failure.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain

