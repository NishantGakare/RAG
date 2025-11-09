from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("rag_data.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
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

vector_store = Chroma(
    collection_name="Nishant_gakare_information",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
