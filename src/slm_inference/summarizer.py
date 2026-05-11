import json
import os
import re
from llama_cpp import Llama

class TraceSummarizer:
    def __init__(self, logs_path: str):
        """
        Inicializa el Summarizer cargando los logs dinámicos para poder "hidratar"
        las trazas estáticas.
        """
        self.logs_path = logs_path
        self.events_by_node = self._load_events()

    def _load_events(self):
        """
        Lee el archivo JSONL de logs y agrupa los mensajes por node_id.
        Esto permite una búsqueda ultra rápida al hidratar la traza.
        """
        events = {}
        if not os.path.exists(self.logs_path):
            print(f"[!] Aviso: No se encontraron logs en {self.logs_path}")
            return events

        with open(self.logs_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    source = json.loads(line)
                    node_info = source.get("node") or source.get("sourceNode")
                    node_id = None
                    if isinstance(node_info, dict):
                        node_id = node_info.get("id")
                    elif isinstance(node_info, str):
                        node_id = node_info
                        
                    msg = source.get("msg", "")
                    
                    if node_id and msg and "Connect Timeout Error" not in str(msg):
                        # Limpiar un poco el msg
                        clean_msg = str(msg).strip()[:200] # Tomamos max 200 chars por evento
                        if node_id not in events:
                            events[node_id] = []
                        # Evitar duplicados exactos para no inflar el prompt
                        if clean_msg not in events[node_id]:
                            events[node_id].append(clean_msg)
                except Exception:
                    continue
        return events

    def hydrate_trace(self, static_trace: dict) -> str:
        """
        Toma una Traza Estática y la combina con muestras de logs reales.
        Retorna el Prompt Maestro listo para el SLM.
        """
        trace_id = static_trace.get("id", "Unknown")
        nodos = static_trace.get("Nodos", [])
        
        prompt_parts = []
        prompt_parts.append(f"TRACE ID: {trace_id}")
        prompt_parts.append("SEQUENCE OF NODES AND LOG EXAMPLES:")
        
        for i, nodo_dict in enumerate(nodos):
            # Formato esperado: {"Nombre del Nodo": "id_del_nodo"}
            for node_name, node_id in nodo_dict.items():
                step_info = f"Step {i+1}: [{node_name}] (ID: {node_id})"
                
                # Buscar muestras de logs para este nodo
                log_samples = self.events_by_node.get(node_id, [])
                if log_samples:
                    # Tomar máximo 3 muestras para dar contexto sin saturar la ventana
                    samples = " | ".join(log_samples[:3])
                    step_info += f"\n  - Real execution log: \"{samples}\""
                else:
                    step_info += f"\n  - Real execution log: (No dynamic data available)"
                    
                prompt_parts.append(step_info)
                
        return "\n".join(prompt_parts)

    def generate_master_prompt(self, static_trace: dict) -> tuple:
        """
        Construye el prompt final con las instrucciones para el SLM.
        Retorna (system_prompt, user_prompt) para usar con Chat Completion.
        """
        hydrated_data = self.hydrate_trace(static_trace)
        
        system_prompt = """You are a Senior Business Analyst and Software Architect at NTT DATA.
Your task is to analyze a "Trace" (an execution path of a Node-RED workflow) and deduce its business use case.
The Trace contains a sequence of nodes (some with technical names) and real execution logs that show what data was processed.

Based on this technical sequence and the log samples, provide a business description of what this workflow path achieves.
Your description should sound like an item in a corporate service catalog (e.g., "Generates an audit report from documents" or "Authenticates a user and queries Okta").

Respond ONLY with a valid JSON using this strict format:
{
  "useCaseName": "A short, descriptive name (max 5 words)",
  "businessContext": "A clear, 1-2 sentence description of the business purpose and action performed."
}
"""
        
        user_prompt = f"""[INPUT TRACE DATA]
{hydrated_data}

[INSTRUCTION]
Analyze the sequence and logs above. Return the JSON. Do not include markdown formatting or any other text.
"""
        return system_prompt, user_prompt

    def run_inference(self, system_prompt: str, user_prompt: str, model_path: str) -> str:
        """Ejecuta la inferencia local usando una instancia persistente del modelo."""
        if not hasattr(self, 'llm'):
            print(f"[*] Inicializando SLM en memoria desde {os.path.basename(model_path)} (solo se hará una vez)...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
        
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, # Muy baja para asegurar precisión y JSON
            max_tokens=150
        )
        
        return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test rápido de hidratación
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    FLOW_ID = os.getenv("FLOW_ID", "11014")
    logs_path = os.path.join(base_dir, os.getenv("ELK_LOGS_DIR", "data/raw/flows_logs"), f"logs_{FLOW_ID}.jsonl")
    static_traces_path = os.path.join(base_dir, os.getenv("STATIC_TRACES_DIR", "data/processed/static_traces"), f"traces_{FLOW_ID}.json")
    
    summarizer = TraceSummarizer(logs_path)
    
    with open(static_traces_path, 'r', encoding='utf-8') as f:
        traces_data = json.load(f)
        
    caminos = traces_data.get("Caminos", [])
    if caminos:
        print("=== GENERANDO PROMPT MAESTRO PARA TRAZA 1 ===")
        sys_p, user_p = summarizer.generate_master_prompt(caminos[0])
        print(user_p)
        
        model_file = os.path.join(base_dir, os.getenv("MODELS_DIR", "data/models"), os.getenv("SLM_MODEL_NAME", "Phi-3-mini-4k-instruct-Q4_K_M.gguf"))
        
        if os.path.exists(model_file):
            print("\n=== INICIANDO INFERENCIA SLM ===")
            output = summarizer.run_inference(sys_p, user_p, model_file)
            print("\n=== RESPUESTA DEL MODELO ===")
            print(output)
            
            # Intento de parsear JSON para verificar
            try:
                # Limpiar por si el modelo añadió markdown ```json ... ```
                clean_json = re.sub(r"```json|```", "", output).strip()
                data = json.loads(clean_json)
                print("\n[OK] JSON parseado correctamente:")
                print(f"Name: {data.get('useCaseName')}")
                print(f"Context: {data.get('businessContext')}")
            except Exception as e:
                print(f"\n[!] Error parseando JSON: {e}")
        else:
            print(f"\n[!] No se encontró el modelo en {model_file}. Descárgalo primero.")
