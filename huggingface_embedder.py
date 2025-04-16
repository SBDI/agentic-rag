"""Hugging Face API-based embedder for text embeddings."""

from typing import List, Optional, Union
import os
import time
import requests
from requests.exceptions import RequestException

class HuggingFaceEmbedder:
    """Embedder using Hugging Face Inference API.
    
    This class provides an interface compatible with Agno's embedder classes
    but uses the Hugging Face API instead of local models to save disk space.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        dimensions: Optional[int] = 1024,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        """Initialize the embedder with Hugging Face API configuration.
        
        Args:
            model_name: Name of the model to use on Hugging Face.
                Default is "BAAI/bge-large-en-v1.5" (1024 dimensions).
            dimensions: Expected embedding dimensions (for validation).
                If None, no validation is performed.
            api_key: Hugging Face API key. If None, will try to get from
                environment variable HUGGINGFACE_API_KEY.
            api_url: Custom API URL if needed. If None, will use the standard
                Hugging Face Inference API endpoint.
        """
        self.model_name = model_name
        self.dimensions = dimensions
        
        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Hugging Face API key is required. Either pass it as api_key parameter "
                "or set the HUGGINGFACE_API_KEY environment variable."
            )
        
        # Set up API URL
        self.api_url = api_url or f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Test connection to validate API key and model availability
        self._test_connection()
    
    def _test_connection(self):
        """Test the connection to the Hugging Face API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            # Send a small test request
            response = requests.post(
                self.api_url,
                headers=headers,
                json={"inputs": "This is a test sentence."},
            )
            
            if response.status_code == 200:
                # If successful, check dimensions if specified
                if self.dimensions is not None:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        actual_dims = len(result[0])
                        if actual_dims != self.dimensions:
                            raise ValueError(
                                f"Model {self.model_name} produces embeddings with {actual_dims} dimensions, "
                                f"but {self.dimensions} were requested."
                            )
            else:
                error_msg = f"API test failed with status code {response.status_code}"
                try:
                    error_details = response.json()
                    error_msg += f": {error_details}"
                except:
                    error_msg += f": {response.text}"
                raise ValueError(error_msg)
                
        except RequestException as e:
            raise ConnectionError(f"Failed to connect to Hugging Face API: {str(e)}")
    
    def _make_api_request(self, texts, max_retries=3, retry_delay=2):
        """Make a request to the Hugging Face API with retry logic."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json={"inputs": texts},
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Too Many Requests
                    # Wait longer for rate limit errors
                    time.sleep(retry_delay * 2 * (attempt + 1))
                    continue
                else:
                    error_msg = f"API request failed with status code {response.status_code}"
                    try:
                        error_details = response.json()
                        error_msg += f": {error_details}"
                    except:
                        error_msg += f": {response.text}"
                    raise ValueError(error_msg)
                    
            except RequestException as e:
                if attempt == max_retries - 1:
                    raise ConnectionError(f"Failed to connect to Hugging Face API after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay * (attempt + 1))
    
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the given text(s) using Hugging Face API.
        
        Args:
            texts: A single text string or a list of text strings to embed.
            
        Returns:
            If a single text is provided, returns a single embedding vector as a list of floats.
            If multiple texts are provided, returns a list of embedding vectors.
        """
        # For BGE models, use the specific prompt template
        if isinstance(texts, str):
            # Single text
            text_to_embed = f"Represent this sentence for searching relevant passages: {texts}"
            embeddings = self._make_api_request([text_to_embed])
            return embeddings[0]
        else:
            # List of texts
            texts_to_embed = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
            # Process in batches if there are many texts to avoid API limits
            if len(texts_to_embed) > 10:
                # Process in batches of 10
                all_embeddings = []
                for i in range(0, len(texts_to_embed), 10):
                    batch = texts_to_embed[i:i+10]
                    batch_embeddings = self._make_api_request(batch)
                    all_embeddings.extend(batch_embeddings)
                return all_embeddings
            else:
                return self._make_api_request(texts_to_embed)
    
    def get_dimensions(self) -> int:
        """Get the dimensions of the embeddings produced by this model."""
        if self.dimensions is not None:
            return self.dimensions
        
        # If dimensions weren't specified, get them by making a test embedding
        test_embedding = self.embed("Test sentence")
        return len(test_embedding)
        
    def id(self) -> str:
        """Return the model ID for compatibility with Agno."""
        return f"huggingface-api:{self.model_name}"
