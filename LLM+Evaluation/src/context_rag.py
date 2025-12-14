from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from config import settings
from sqlalchemy import text
import shutil
import os
from src.evaluation_metrics import load_data_prepare_test_case

EMBEDDING_MODEL = settings.EMBEDDING_MODEL
PERSIST_DIR = settings.PERSIST_DIR
user_query = ""
ai_response = ""

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

def build_context_rag( chat_json_path, context_json_path, user_turn, ai_turn, persist_dir=PERSIST_DIR):
    global user_query, ai_response
    """Embed context into Chroma using LangChain."""
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(collection_name="schema_context", embedding_function=embedding_model, persist_directory=persist_dir)
    docs = []
    data = load_data_prepare_test_case(chat_json_path, context_json_path, user_turn, ai_turn)
    user_query = data["user_query"]
    ai_response = data["actual_output"]
    retrieval_context = data["context_vectors"]
    for i in retrieval_context:
            docs.append(i)
    vectorstore.add_texts(docs)
    # vectorstore._client.persist()
    print(f"Stored {len(docs)} schema entries in Chroma vector store.")
    return vectorstore
