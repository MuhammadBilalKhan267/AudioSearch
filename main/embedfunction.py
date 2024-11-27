from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_community.embeddings import OllamaEmbeddings

class Embedder(EmbeddingFunction):
    def __init__(self, model):
        self.model = OllamaEmbeddings(model=model)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.embed_documents(input)
        return embeddings
    
    def embed_query(self, query):
        return self.model.embed_query(query)
    
    def embed_documents(self, document):
        return self.model.embed_documents(document)