from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from pathlib import Path
import PyPDF2  # pip install PyPDF2

#SETTINGS
CSV_PATH = "rag_data.csv"
PDF_DIR = Path("data/pdfs")
TXT_DIR = Path("data/txts")
DB_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "Nishant_gakare_information"
EMBED_MODEL = "mxbai-embed-large"
CHUNK_SIZE = 800   # characters per chunk (tuneable)
CHUNK_OVERLAP = 100
# -----------

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple character-based chunker. Returns list of text chunks."""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def load_csv_docs(csv_path):
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        text = f"Topic: {row['topic']}\nInformation: {row['information']}"
        #chunk if too long
        for i, chunk in enumerate(chunk_text(text)):
            doc = Document(
                page_content=chunk,
                metadata={"source": "csv", "topic": row["topic"], "id": str(row["id"])},
                id=f"csv-{row['id']}-{i}"
            )
            documents.append(doc)
    return documents

def load_pdf_docs(pdf_dir):
    documents = []
    if not pdf_dir.exists():
        return documents
    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            reader = PyPDF2.PdfReader(str(pdf_path))
            text = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
            full_text = "\n".join(text)
            for i, chunk in enumerate(chunk_text(full_text)):
                doc = Document(
                    page_content=chunk,
                    metadata={"source": str(pdf_path.name)},
                    id=f"pdf-{pdf_path.stem}-{i}"
                )
                documents.append(doc)
        except Exception as e:
            print(f"Failed to read {pdf_path}: {e}")
    return documents

def load_txt_docs(txt_dir):
    documents = []
    if not txt_dir.exists():
        return documents
    for txt_path in txt_dir.glob("*.txt"):
        text = txt_path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            doc = Document(
                page_content=chunk,
                metadata={"source": str(txt_path.name)},
                id=f"txt-{txt_path.stem}-{i}"
            )
            documents.append(doc)
    return documents

#vector store setup
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
add_documents = not os.path.exists(DB_DIR)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_DIR,
    embedding_function=embeddings,
)

if add_documents:
    all_docs = []
    all_docs += load_csv_docs(CSV_PATH)
    all_docs += load_pdf_docs(PDF_DIR)
    all_docs += load_txt_docs(TXT_DIR)

    if all_docs:
        ids = [d.id for d in all_docs]
        print(f"Adding {len(all_docs)} docs to Chroma...")
        vector_store.add_documents(documents=all_docs, ids=ids)
    else:
        print("No documents found to add. Check CSV/PDF/TXT paths.")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
