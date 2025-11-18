from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)
from sentence_transformers import SentenceTransformer
import uuid


class Retriever:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="rag_collection",
        vector_size=None,
        db_path="qdrant_data",
    ):
        """
        Initialize a retriever using Qdrant as the vector store.

        Args:
            model_name (str): Embedding model.
            collection_name (str): Name of the Qdrant collection.
            vector_size (int): Dimension of embedding vectors.
            db_path (str): Path to Qdrant local directory.
        """
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name

        self.client = QdrantClient(path=db_path)
        self.vector_size = vector_size

        if not self.client.collection_exists(collection_name) and vector_size:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def _ensure_collection(self, embeddings):
        """
        Automatically create the Qdrant collection on first insert.
        """
        if self.vector_size is None:
            self.vector_size = embeddings.shape[1]

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )

    def add_documents(self, docs):
        """
        Add documents to Qdrant with their embeddings.

        Args:
            docs (list[str]): Text documents to index.
        """
        if not docs:
            return

        embeddings = self.model.encode(docs).astype("float32")
        self._ensure_collection(embeddings)

        points = []
        for doc, emb in zip(docs, embeddings):
            points.append(
                PointStruct(id=str(uuid.uuid4()), vector=emb, payload={"text": doc})
            )

        self.client.upsert(collection_name=self.collection_name, points=points)


    def retrieve(self, query, k=3):
        """
        Retrieve the top-k most similar documents using Qdrant (embedded mode).
        """
        if not self.client.collection_exists(self.collection_name):
            raise ValueError("Qdrant collection does not exist. Add documents first.")

        q_emb = self.model.encode(query).astype("float32")

        results = self.client.query_points(
            collection_name=self.collection_name, query=q_emb.tolist(), limit=k
        ).points

        return [p.payload["text"] for p in results]
