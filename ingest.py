import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

DATA_PATH = "data"
VECTOR_PATH = "vectorstore/faiss_index"


def _get_embedding_device():
    """Auto-detect GPU if available."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _row_to_text(row, dataset_name):
    """Convert a CSV row to readable text for embedding."""
    items = []
    for key, value in row.items():
        if pd.isna(value):
            continue
        if isinstance(value, (int, float)):
            items.append(f"{key}: {value}")
        else:
            items.append(f"{key}: {str(value)[:200]}")  # Truncate long values
    return f"[{dataset_name}] " + " | ".join(items)


def ingest_documents():
    """Ingest all CSVs from data folder into FAISS vector store."""
    documents = []
    total_rows = 0

    print("📂 Scanning data folder...")
    
    for root, _, files in os.walk(DATA_PATH):
        for file in sorted(files):
            if not file.lower().endswith(".csv"):
                continue

            file_path = os.path.join(root, file)
            dataset_name = os.path.splitext(file)[0]
            
            try:
                df = pd.read_csv(file_path, low_memory=False)
                print(f"\n📄 Processing: {file} ({len(df)} rows)")
                
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Indexing {dataset_name}"):
                    text = _row_to_text(row, dataset_name)
                    
                    metadata = {
                        "dataset": dataset_name,
                        "file": os.path.relpath(file_path, DATA_PATH),
                        "row_index": int(idx),
                    }
                    
                    # Extract key columns if they exist for easier filtering
                    if "Name" in df.columns and not pd.isna(row.get("Name")):
                        metadata["name"] = str(row.get("Name"))
                    if "Sector" in df.columns and not pd.isna(row.get("Sector")):
                        metadata["sector"] = str(row.get("Sector"))
                    if "Years of Operation" in df.columns and not pd.isna(row.get("Years of Operation")):
                        metadata["years"] = str(row.get("Years of Operation"))

                    documents.append(Document(page_content=text, metadata=metadata))
                    total_rows += 1
            
            except Exception as e:
                print(f"  ⚠️  Error reading {file}: {e}")
                continue

    if not documents:
        print("❌ No documents found to index!")
        return

    print(f"\n🔄 Creating embeddings for {total_rows} documents...")
    print(f"🖥️  Device: {_get_embedding_device()}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": _get_embedding_device()},
    )
    
    print("⏳ This may take a few minutes on CPU...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTOR_PATH)

    print(f"\n✅ Successfully indexed {total_rows} documents!")
    print(f"📦 Vector store saved to: {VECTOR_PATH}")


if __name__ == "__main__":
    ingest_documents()
