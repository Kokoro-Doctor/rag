from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectorstore(path: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=path, embedding_function=embeddings)

def get_vectorstores():
    heart_db = load_vectorstore("/mnt/efs/chroma_db")
    gyno_db = load_vectorstore("/mnt/efs/gyno_db")
    return {"heart": heart_db, "gyno": gyno_db}
