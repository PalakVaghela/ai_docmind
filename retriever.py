from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# we are writing this again because, we have stored the embedings in chroma_db so the quetion need to convert into the same model type in which data is
# also question should retirvie the ans od question from chroma_db path

def get_embeddingds():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# it takes text and convert it into vector or embeddings.

def get_vector_store():
    embeddings = get_embeddingds()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

# here k=3 is used to tell that this no. of relevent chunk should it show
def retrive_document(query):
    vector_store = get_vector_store()
    results = vector_store.similarity_search(
        query,
        k=3
    )
    return results

if __name__ == "__main__":
    query = "In which column in gernal ledger we have to make a change?"
    results = retrive_document(query)
    for index,doc in enumerate(results, start=1):
        print(f"Result {index}:")
        print(doc.page_content)
        print("-" * 50)
