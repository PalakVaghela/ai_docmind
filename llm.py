from langchain_ollama import ChatOllama

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

if __name__ == "__main__":
    prompt = "Explain what FastAPI is in simple words."
    response = call_llm(prompt)
    print("\nOllama response:")
    print(response)
