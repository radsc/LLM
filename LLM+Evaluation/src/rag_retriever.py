from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from config import settings

PERSIST_DIR = settings.PERSIST_DIR

# Initialize and return a retriever for schema context
def get_retriever(persist_dir=PERSIST_DIR):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="schema_context",
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever
