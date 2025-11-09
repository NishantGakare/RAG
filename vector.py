from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load your CSV
df = pd.read_csv("rag_data.csv")

# Initialize embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Combine topic + information into one text block
        content = f"Topic: {row['topic']}\nInformation: {row['information']}"
        document = Document(
            page_content=content,
            metadata={
                "source": "rag_data.csv",
                "topic": row["topic"],
                "id": row["id"]
            },
            id=str(row["id"])
        )
        ids.append(str(row["id"]))
        documents.append(document)

# Create or load the Chroma vector store
vector_store = Chroma(
    collection_name="Nishant_gakare_information",
    persist_directory=db_location,
    embedding_function=embeddings,
)

# Add documents only the first time
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Make retriever for queries
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
