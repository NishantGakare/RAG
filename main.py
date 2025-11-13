from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a highly knowledgeable AI assistant that knows detailed information about Nishant Gakare (also called RN).

You are given the following context (retrieved information):
{information}

Answer the user's question in a friendly and detailed way using ONLY this context.
If the context doesn’t contain the answer, say:
"I don’t have enough information about that yet."

When you respond:
- Write complete, natural sentences.
- If possible, mention where the info came from (e.g., his project, event, or friends).
- Avoid repeating the question.

User's Question:
{question}
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("🧠 KnowRN-RAG Chatbot is ready! (type 'exit' to quit)\n")

while True:
    question = input("Ask something about Nishant Gakare: ")
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    docs = retriever.invoke(question)

    print("\n📚 Retrieved Sources:")
    for d in docs:
        src = d.metadata.get("source", "unknown")
        title = d.metadata.get("file_name", d.metadata.get("topic", "N/A"))
        print(f" - Source: {src}, File/Topic: {title}")

    information = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])

    result = chain.invoke({"information": information, "question": question})

    print("\n🤖 Answer:", result, "\n")

