import os
from pathlib import Path
import pandas as pd
import PyPDF2
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize LLM for reranking
llm = OllamaLLM(model="phi")

# Paths and constants
BACKEND_DIR = Path(__file__).parent
CSV_PATH = BACKEND_DIR / "rag_data.csv"
PDF_DIR = BACKEND_DIR / "data" / "pdfs"
TXT_DIR = BACKEND_DIR / "data" / "txts"
DB_DIR = BACKEND_DIR / "chroma_langchain_db"
COLLECTION_NAME = "Nishant_gakare_information"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# -----------------------------
# Helper: Chunk text
# -----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "],
        length_function=len,
    )
    return splitter.split_text(text)

# -----------------------------
# Load CSV documents
# -----------------------------
def load_csv_docs(csv_path):
    documents = []
    if not os.path.exists(csv_path):
        print(f"⚠️ CSV not found: {csv_path}")
        return documents

    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            text = f"Topic: {row['topic']}\nInformation: {row['information']}"
            for i, chunk in enumerate(chunk_text(text)):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": "csv",
                        "file_name": row["topic"],
                        "id": str(row["id"])
                    },
                    id=f"csv-{row['id']}-{i}"
                )
                documents.append(doc)
        print(f"✅ Loaded {len(documents)} CSV chunks")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
    return documents

# -----------------------------
# Load PDF documents
# -----------------------------
def load_pdf_docs(pdf_dir):
    documents = []
    if not pdf_dir.exists():
        print(f"⚠️ PDF directory not found: {pdf_dir}")
        return documents

    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            reader = PyPDF2.PdfReader(str(pdf_path))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            for i, chunk in enumerate(chunk_text(text)):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": "pdf",
                        "file_name": pdf_path.name
                    },
                    id=f"pdf-{pdf_path.stem}-{i}"
                )
                documents.append(doc)
        except Exception as e:
            print(f"❌ Failed to read {pdf_path}: {e}")
    print(f"✅ Loaded {len(documents)} PDF chunks")
    return documents

# -----------------------------
# Load TXT documents
# -----------------------------
def load_txt_docs(txt_dir):
    documents = []
    if not txt_dir.exists():
        print(f"⚠️ TXT directory not found: {txt_dir}")
        return documents

    for txt_path in txt_dir.glob("*.txt"):
        try:
            text = txt_path.read_text(encoding="utf-8")
            for i, chunk in enumerate(chunk_text(text)):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": "txt",
                        "file_name": txt_path.name
                    },
                    id=f"txt-{txt_path.stem}-{i}"
                )
                documents.append(doc)
        except Exception as e:
            print(f"❌ Failed to read {txt_path}: {e}")
    print(f"✅ Loaded {len(documents)} TXT chunks")
    return documents

# -----------------------------
# Initialize embeddings and vector store
# -----------------------------
print("🧠 Initializing embeddings and vector store...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_DIR,
    embedding_function=embeddings,
)

# Load all documents
csv_docs = load_csv_docs(CSV_PATH)
pdf_docs = load_pdf_docs(PDF_DIR)
txt_docs = load_txt_docs(TXT_DIR)
all_docs = csv_docs + pdf_docs + txt_docs

print(f"\n📊 Summary:")
print(f"   CSV docs: {len(csv_docs)}")
print(f"   PDF docs: {len(pdf_docs)}")
print(f"   TXT docs: {len(txt_docs)}")
print(f"   Total chunks to add/update: {len(all_docs)}")

if all_docs:
    ids = [d.id for d in all_docs]
    print(f"📚 Adding or updating {len(all_docs)} chunks in Chroma...")
    vector_store.add_documents(documents=all_docs, ids=ids)
    print("✅ Chroma database updated successfully!")
else:
    print("⚠️ No documents found. Please check your paths and file types.")

# -----------------------------
# Retriever (use MMR for diversity across sources)
# -----------------------------
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,       # final results returned
        "fetch_k": 20 # pool size before MMR diversification
    }
)
print("🚀 Retriever (MMR) ready to use!")

# -----------------------------
# Reranker for Stage 2
# -----------------------------
def rerank_docs(question, docs, top_k=3):
    """
    Simple reranker that uses Ollama LLM to score how relevant each doc is to the question.
    Returns top_k documents.
    """
    rerank_llm = OllamaLLM(model="phi")
    scored_docs = []

    for doc in docs:
        prompt = f"""
        Question: {question}
        Context: {doc.page_content}

        Score how relevant this context is to the question on a scale of 0 (irrelevant) to 10 (very relevant).
        Only return the number.
        """
        try:
            score_str = rerank_llm.invoke(prompt).strip()
            score = float(score_str)
        except:
            score = 0
        scored_docs.append((score, doc))

    # Sort by score descending and take top_k
    ranked = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[:top_k]]
    return top_docs

# -----------------------------
# Test query if run directly
# -----------------------------
if __name__ == "__main__":
    query = input("\n🔍 Test query: ")
    results = retriever.invoke(query)
    print(f"\nTop {len(results)} results:")
    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:400])
        print("📁 Source:", doc.metadata.get("file_name") or doc.metadata.get("source"))
