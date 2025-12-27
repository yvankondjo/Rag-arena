from pathlib import Path
from typing import List, Dict
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from ragbench.config import Config
import tiktoken
class ChromaDBIndex:
    def __init__(
        self, 
        collection_name: str = Config.VECTOR_STORE_COLLECTION,
        persist_directory: str = Config.PERSIST_DIRECTORY,
        embedding_model: str = Config.EMBEDDING_MODEL,
        threshold: float = 0.75
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embedding_model
        )
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        self.threshold = threshold
        try:
            self.encoding = tiktoken.encoding_for_model(embedding_model)
        except KeyError:
            try:
                self.encoding = tiktoken.encoding_for_model("o200k_base")
            except KeyError:
                self.encoding = tiktoken.get_encoding("o200k_base")

    
    def delete_document(self, document_id: str) -> bool:
        """
        Supprime tous les chunks d'un document
        
        Args:
            document_id: L'identifiant du document à supprimer
        
        Returns:
            True si des chunks ont été supprimés, False sinon
        """
        try:
            results = self.collection.delete(where={"document_id": document_id})
            return True
        except Exception as e:
            print(f"Erreur lors de la suppression du document {document_id}: {e}")
            return False
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """
        Récupère les chunks et leurs métadonnées à partir de leurs IDs
        
        Args:
            chunk_ids: Liste des IDs des chunks à récupérer
            
        Returns:
            Liste de dicts avec 'id', 'text', 'metadata'
        """
        if not chunk_ids:
            return []
        
        try:
            results = self.collection.get(ids=chunk_ids)
            chunks = []
            
            if results.get('ids'):
                ids = results['ids']
                documents = results.get('documents', [])
                metadatas = results.get('metadatas', [])
                
                for i, chunk_id in enumerate(ids):
                    chunks.append({
                        'id': chunk_id,
                        'text': documents[i] if i < len(documents) else '',
                        'metadata': metadatas[i] if i < len(metadatas) else {}
                    })
            
            return chunks
        except Exception as e:
            return []

