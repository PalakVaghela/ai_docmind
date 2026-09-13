from langchain_ollama import ChatOllama
from retriever import retrive_document

MODEL_NAME = "llama3.2"

def get_llm():
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

def call_llm(prompt: str):
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content

def answer_question(query: str):
    results = retrive_document(query)

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for index, (document, score) in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"Score: {score}")
        print(document.page_content)

    context = "\n\n".join(
        document.page_content
        for document, score in results
    )

    print("\n" + "=" * 60)
    print("FINAL CONTEXT SENT TO LLM")
    print("=" * 60)
    print(context)

    prompt = f"""
You are a helpful document assistant.

Answer the user's question using the provided context.

Rules:
- Use only the information present in the context.
- Do not make up information.
- If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    return call_llm(prompt)

if __name__ == "__main__":
    prompt = "In which column in gernal ledger we have to add?"
    # prompt = "Who is best friend of doraemon in japanees cartoon doraemon?"
    response = call_llm(prompt)
    print("\nOllama response=========================================")
    print(response)

    ans = answer_question(prompt)
    print("\nAnswer:================================================")
    print(ans)

# currenlty the llm is independent it is not taking context of document now we have to add that.
# and currenlty the we are only getting relevant chunks only.
