from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class MemoryDatabase:
    def __init__(self, initial_memories: list[str]):
        self.index = VectorStoreIndex.from_documents(
            [Document(text=memory) for memory in initial_memories],
            embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        )

    def add_memories(self, memories: list[str]):
        for memory in memories:
            self.index.insert(Document(text=memory))
    
    def retrieve(self, query: str, top_k: int = 5):
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        return retriever.retrieve(query)
