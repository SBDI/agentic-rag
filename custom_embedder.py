"""Custom embedder using BGE Large model for open-source embedding."""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from requests.exceptions import ChunkedEncodingError

class BGEEmbedder:
    """Embedder using BGE Large model.
    
    This class provides an interface compatible with Agno's embedder classes
    but uses the open-source BGE Large model instead of OpenAI.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        dimensions: Optional[int] = 1024,
        device: str = "cpu",
    ):
        """Initialize the embedder with the BGE Large model.
        
        Args:
            model_name: Name of the model to use.
                Default is "BAAI/bge-large-en-v1.5" (1024 dimensions).
            dimensions: Expected embedding dimensions (for validation).
                If None, no validation is performed.
            device: Device to run the model on ("cpu" or "cuda").
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.model = SentenceTransformer(model_name, device=device)
                break
            except (ChunkedEncodingError, Exception) as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to load model after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay * (attempt + 1))
        
        self.model_name = model_name
        self.dimensions = dimensions
        self.device = device
        
        # Validate dimensions if specified
        if dimensions is not None:
            actual_dims = self.model.get_sentence_embedding_dimension()
            if actual_dims != dimensions:
                raise ValueError(
                    f"Model {model_name} produces embeddings with {actual_dims} dimensions, "
                    f"but {dimensions} were requested."
                )
    
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the given text(s).
        
        Args:
            texts: A single text string or a list of text strings to embed.
            
        Returns:
            If a single text is provided, returns a single embedding vector as a list of floats.
            If multiple texts are provided, returns a list of embedding vectors.
        """
        # BGE models work best with a specific prompt template
        if isinstance(texts, str):
            # Single text
            text_to_embed = f"Represent this sentence for searching relevant passages: {texts}"
            embedding = self.model.encode(text_to_embed)
            return embedding.tolist()
        else:
            # List of texts
            texts_to_embed = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
            embeddings = self.model.encode(texts_to_embed)
            return [emb.tolist() for emb in embeddings]
    
    def get_dimensions(self) -> int:
        """Get the dimensions of the embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()
        
    def id(self) -> str:
        """Return the model ID for compatibility with Agno."""
        return self.model_name
