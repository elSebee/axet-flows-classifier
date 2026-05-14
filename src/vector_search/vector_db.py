import json
import pickle
import os
import numpy as np
import httpx
import os
os.environ["CURL_CA_BUNDLE"] = ""
original_init = httpx.Client.__init__
def new_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = new_init

from typing import List, Dict, Tuple
from fastembed import TextEmbedding

class VectorDatabase:
    def __init__(self, db_path: str = None):
        """
        Inicializa la base de conocimiento vectorial.
        Usa modelo BGE-Small por defecto para evitar saturar la RAM.
        """
        self.db_path = db_path
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
        self.embedding_model = TextEmbedding(model_name=model_name)
        self.vectors = None
        self.metadata = []

    def build_knowledge_base(self, use_cases_json_path: str, save_path: str):
        """
        Lee el catálogo de Casos de Uso, extrae intenciones semánticas,
        genera los embeddings y los guarda a disco.
        """
        with open(use_cases_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents_to_embed = []
        self.metadata = []

        # 1. Normalización del Catálogo
        for area in data:
            area_code = area.get("code", "")
            area_name = area.get("displayName", "")
            
            for uc in area.get("useCases", []):
                # Solo indexar casos de uso activos
                if not uc.get("enabled", True):
                    continue

                uc_code = uc.get("useCaseCode", "")
                display_name = uc.get("displayName", "")
                description = uc.get("description", "")
                
                # Fórmula del "Espacio de Intenciones"
                text_to_embed = f"[{area_name}] {display_name}. Contexto: {description}"
                
                documents_to_embed.append(text_to_embed)
                
                # Guardamos la metadata para referenciar cuando el kNN devuelva un índice
                self.metadata.append({
                    "useCaseCode": uc_code,
                    "displayName": display_name,
                    "area": area_name,
                    "description": description,
                    "embedded_text": text_to_embed
                })

        print(f"[*] Generando embeddings para {len(documents_to_embed)} Casos de Uso con BGE-Small...")
        
        # 2. Generar Embeddings (FastEmbed usa ONNX, muy rápido en CPU)
        # embedding_model.embed() devuelve un generador, lo convertimos a lista y luego a numpy array
        embeddings_generator = self.embedding_model.embed(documents_to_embed)
        self.vectors = np.vstack(list(embeddings_generator))

        print(f"[*] ¡Embeddings generados exitosamente! Shape: {self.vectors.shape}")

        # 3. Guardar la Base de Datos Local
        db_data = {
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(db_data, f)
            
        print(f"[*] Base de Conocimiento guardada en {save_path}")

    def load(self, path: str):
        """Carga la base de vectores y metadatos desde disco."""
        with open(path, 'rb') as f:
            db_data = pickle.load(f)
        self.vectors = db_data["vectors"]
        self.metadata = db_data["metadata"]

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Calcula Similitud Coseno entre una Traza Canónica (Query) 
        y la base de Casos de Uso. Devuelve los Top K.
        """
        if self.vectors is None:
            raise ValueError("La Base Vectorial no ha sido cargada o construida.")

        # Generar embedding del query
        query_embedding = list(self.embedding_model.embed([query]))[0]
        
        # Similitud Coseno (Dot product asumiendo vectores normalizados por fastembed)
        # FastEmbed suele devolver vectores normalizados, por lo que Dot Product == Cosine Similarity
        similarities = np.dot(self.vectors, query_embedding)
        
        # Obtener los top_k índices con mayor similitud
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            res = self.metadata[idx].copy()
            res["score"] = float(similarities[idx])
            results.append(res)
            
        return results

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Script ejecutable para construir la DB inicial
    print("=== aXet.flows Classifier: Generador de Vectores ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    use_cases_path = os.path.join(base_dir, os.getenv("USE_CASES_FILE", "data/raw/use_cases/useCases.json"))
    db_save_path = os.path.join(base_dir, os.getenv("PROCESSED_DIR", "data/processed"), "usecases_db.pkl")
    
    db = VectorDatabase()
    db.build_knowledge_base(use_cases_path, db_save_path)
