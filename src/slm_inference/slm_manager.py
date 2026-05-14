import os
from typing import Optional
from llama_cpp import Llama

class SLMManager:
    """
    Centralized manager for the SLM model lifecycle.
    Ensures the model is loaded only once and shares the instance.
    """

    DEFAULT_N_CTX = 2560
    DEFAULT_N_BATCH = 512
    DEFAULT_N_THREADS = 4

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._llm: Optional[Llama] = None

    @property
    def llm(self) -> Llama:
        """Lazy-load the SLM on first use."""
        if self._llm is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"SLM model not found at {self.model_path}")
                
            print(f"  [SLMManager] Loading SLM: {os.path.basename(self.model_path)}...")
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.DEFAULT_N_CTX,
                n_batch=self.DEFAULT_N_BATCH,
                n_threads=self.DEFAULT_N_THREADS,
                verbose=False,
            )
        return self._llm
