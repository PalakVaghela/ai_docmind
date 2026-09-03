from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH = Path("data")
CHROMA_PATH = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    documents = []
    for file_path in DATA_PATH.glob("*.txt"):
        loader = TextLoader(str(file_path))
        documents.extend(loader.load())
    return documents

def split_document(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    # in chunk we have overlap chunks while cut the file in chunks it takes some texts in both the chunks so some are overlaped so that sentence's context get remian/
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    return vector_store

if __name__ == "__main__":
    print("Loading documents ....")
    documents = load_documents()

    print(f"Loaded a {len(documents)} document(s)")
    print("Splitting document")

    chunks = split_document(documents)
    print(f"Created {len(chunks)} number of chunks")

    print("Creating embeddings and storing in ChromaDB...")

    create_vector_store(chunks)
    print("Done! Documents successfully stored in ChromaDB.")



# we create a document -> it is splitted into chunks -> then gives to chroma db -> chroma db converts it into embedding
# so when user asks a questions then it is converted into embeddings and then it check that which chunks embedding matches with question.
# then it answers the question based on that chunk.
