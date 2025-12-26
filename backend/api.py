from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vector import retriever, rerank_docs  # make sure rerank_docs is in vector.py
from langchain_ollama import OllamaLLM

app = FastAPI(title="KnowRN-RAG Chatbot Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "KnowRN-RAG backend running ✅"}

@app.post("/query")
async def query_rag(data: Query):
    question = data.question
    initial_docs = retriever.invoke(question)
    top_docs = rerank_docs(question, initial_docs, top_k=3)
    context = "\n\n".join([doc.page_content for doc in top_docs])
    llm = OllamaLLM(model="phi")
    prompt = f"""
You are a knowledgeable assistant.

Context:
{context}

Question: {question}
Answer the question using ONLY the context. If not in context, reply: "I don't have enough information about that yet."
"""
    answer = llm.invoke(prompt)
    #Collect sources
    sources = [
        doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
        for doc in top_docs
    ]
    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
