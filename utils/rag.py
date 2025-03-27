from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import settings


class RAGRetriever:
    def __init__(self):
        # self.embeddings = HuggingFaceEmbeddings(model_name=config.settings.EMBEDDING_MODEL)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}  # 使用CPU运行
        )
        self.vector_store = None

    def init_vector_store(self, documents):
        docs = [Document(page_content=doc) for doc in documents]
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def retrieve(self, query, k=3):
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)