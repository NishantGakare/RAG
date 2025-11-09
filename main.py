from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a helpful assistant that knows about the user whose name is Nishant Gakare

here is some information about Nishant: {information}

here is the question to answer: {question}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("Enter your question about Nishant Gakare: ")
    if question.lower() in ["exit", "quit"]:
        break

    docs = retriever.invoke(question)
    information = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])

    result = chain.invoke({"information": information, "question": question})
    print(result)