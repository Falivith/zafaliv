import os
import shutil
from qdrant_client import QdrantClient


class VectorStorageManager:
    def __init__(self, db_path="qdrant_data"):
        """
        Manage a local embedded Qdrant database.

        Args:
            db_path (str): Path where Qdrant stores its internal files.
        """
        self.db_path = db_path
        self.client = QdrantClient(path=db_path)

    def list_collections(self):
        """
        Returns a list of all Qdrant collections.
        """
        return [c.name for c in self.client.get_collections().collections]

    def drop_collection(self, collection_name):
        """
        Deletes a single Qdrant collection.
        """
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            return True
        return False

    def collection_info(self, collection_name):
        """
        Returns metadata about a collection.
        """
        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        return self.client.get_collection(collection_name)

    def count_points(self, collection_name):
        """
        Returns the number of stored vectors in the collection.
        """
        if not self.client.collection_exists(collection_name):
            return 0

        result = self.client.count(collection_name)
        return result.count

    def wipe_database(self):
        """
        Completely deletes the entire Qdrant database from disk.
        Use with caution — this removes everything.
        """
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        os.makedirs(self.db_path, exist_ok=True)

    def clean_collection(self, collection_name):
        """
        Deletes and recreates a collection.
        """
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

    def inspect_payloads(self, collection_name, limit=20):
        """
        Fetches points with payloads for inspection.
        """
        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        records, next_page = self.client.scroll(
            collection_name=collection_name,
            limit=limit
        )

        return [{"id": r.id, "payload": r.payload} for r in records]
