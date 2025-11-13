import os
from pathlib import Path
import pandas as pd
import PyPDF2
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CSV_PATH = "rag_data.csv"
PDF_DIR = Path("data/pdfs")
TXT_DIR = Path("data/txts")
DB_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "Nishant_gakare_information"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

#HELPER: CHUNK TEXT
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

#LOAD CSV DOCS
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
                    metadata={"source": "csv", "topic": row["topic"], "id": str(row["id"])},
                    id=f"csv-{row['id']}-{i}"
                )
                documents.append(doc)
        print(f"✅ Loaded {len(documents)} CSV chunks")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

    return documents

#LOAD PDF DOCS
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
                    metadata={"source": str(pdf_path.name)},
                    id=f"pdf-{pdf_path.stem}-{i}"
                )
                documents.append(doc)
        except Exception as e:
            print(f"❌ Failed to read {pdf_path}: {e}")

    print(f"✅ Loaded {len(documents)} PDF chunks")
    return documents

#LOAD TXT DOCS
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
                    metadata={"source": str(txt_path.name)},
                    id=f"txt-{txt_path.stem}-{i}"
                )
                documents.append(doc)
        except Exception as e:
            print(f"❌ Failed to read {txt_path}: {e}")

    print(f"✅ Loaded {len(documents)} TXT chunks")
    return documents

# STORE AND RETRIEVER
print("🧠 Initializing embeddings and vector store...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_DIR,
    embedding_function=embeddings,
)

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

#RETRIEVER
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("🚀 Retriever ready to use!")

if __name__ == "__main__":
    query = input("\n🔍 Test query: ")
    results = retriever.invoke(query)
    print(f"\nTop {len(results)} results:")
    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:400])
        print("📁 Source:", doc.metadata.get("source"))
