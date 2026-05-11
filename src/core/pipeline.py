import json
import os
import sys
import re

# Añadir src al path para importar módulos correctamente
# Añadir root del proyecto al path para importar módulos correctamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from src.graph_engine.tracer import FlowGraphTracer
from src.slm_inference.summarizer import TraceSummarizer
from src.vector_search.vector_db import VectorDatabase

def main():
    print("=== Iniciando Pipeline de AXET.Flows Classifier ===")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    FLOW_ID = os.getenv("FLOW_ID", "11014")
    
    # Rutas desde .env relativas al BASE_DIR
    FLOW_JSON_PATH = os.path.join(BASE_DIR, os.getenv("FLOW_JSON_DIR", "data/raw/flows_data"), f"data_{FLOW_ID}.json")
    ELK_LOGS_PATH = os.path.join(BASE_DIR, os.getenv("ELK_LOGS_DIR", "data/raw/flows_logs"), f"logs_{FLOW_ID}.jsonl")
    STATIC_TRACES_PATH = os.path.join(BASE_DIR, os.getenv("STATIC_TRACES_DIR", "data/processed/static_traces"), f"traces_{FLOW_ID}.json")
    USECASES_DB_PATH = os.path.join(BASE_DIR, os.getenv("PROCESSED_DIR", "data/processed"), "usecases_db.pkl")
    MODEL_PATH = os.path.join(BASE_DIR, os.getenv("MODELS_DIR", "data/models"), os.getenv("SLM_MODEL_NAME", "Phi-3-mini-4k-instruct-Q4_K_M.gguf"))
    OUTPUT_DIR = os.path.join(BASE_DIR, os.getenv("OUTPUT_DIR", "data/output"))
    
    print(f"[*] Trabajando con FLOW_ID: {FLOW_ID}")

    try:
        # Capa 1: Carga de Vector DB
        print("\n[Módulo: VectorSearch] Inicializando base de conocimiento (Capa 2)...")
        if not os.path.exists(USECASES_DB_PATH):
            print(f"[!] Error: La base vectorial no existe en {USECASES_DB_PATH}. Corre vector_db.py primero.")
            return
        
        vector_db = VectorDatabase()
        vector_db.load(USECASES_DB_PATH)
        print(f"[*] VectorDB cargada. {len(vector_db.metadata)} Casos de Uso listos para mapeo.")

        # Capa 2: Extracción de Trazas (Si no existen, las generamos)
        print("\n[Módulo: GraphEngine] Cargando trazas estáticas...")
        if not os.path.exists(STATIC_TRACES_PATH):
            print("[*] No se encontró el archivo de trazas, generándolo ahora...")
            tracer = FlowGraphTracer(FLOW_JSON_PATH)
            tracer.export_paths_to_json(STATIC_TRACES_PATH)
            
        with open(STATIC_TRACES_PATH, 'r', encoding='utf-8') as f:
            traces_data = json.load(f)
        caminos = traces_data.get("Caminos", [])
        print(f"[*] {len(caminos)} Trazas estáticas cargadas.")

        # Capa 3: Hidratación e Inferencia (SLM)
        print("\n[Módulo: SLM Inference] Inicializando summarizer e hidratando trazas...")
        if not os.path.exists(MODEL_PATH):
            print(f"[!] Error: Modelo SLM no encontrado en {MODEL_PATH}.")
            return
            
        summarizer = TraceSummarizer(ELK_LOGS_PATH)
        
        import datetime
        start_time = datetime.datetime.now()
        
        # Procesaremos TODAS las trazas
        trazas_a_procesar = caminos
        print(f"[*] Ejecutando pipeline End-to-End para {len(trazas_a_procesar)} trazas...")
        
        resultados_finales = []
        
        for idx, traza in enumerate(trazas_a_procesar):
            trace_id = traza.get('id', f'T_{idx}')
            print(f"[{idx+1}/{len(trazas_a_procesar)}] Procesando {trace_id}...", end=" ", flush=True)
            
            # 3.1: Generar Prompt
            sys_p, user_p = summarizer.generate_master_prompt(traza)
            
            # 3.2: Inferencia SLM
            output = summarizer.run_inference(sys_p, user_p, MODEL_PATH)
            
            # 3.3: Parsing del JSON
            trace_result = {
                "trace_id": trace_id,
                "nodos": len(traza.get("Nodos", [])),
                "slm_use_case_name": "",
                "slm_business_context": "",
                "top_matches": []
            }
            
            try:
                clean_json = re.sub(r"```json|```", "", output).strip()
                data = json.loads(clean_json)
                business_context = data.get('businessContext', '')
                use_case_name = data.get('useCaseName', '')
                
                trace_result["slm_use_case_name"] = use_case_name
                trace_result["slm_business_context"] = business_context
                
                if business_context:
                    # Capa 4: Semantic Search en Vector DB
                    matches = vector_db.search(business_context, top_k=3) # Top 3 matches
                    trace_result["top_matches"] = matches
                    
                print(f"OK ({len(trace_result['top_matches'])} matches)")
                
            except Exception as e:
                print(f"ERROR: No se pudo parsear el JSON del SLM.")
                trace_result["error"] = str(e)
                trace_result["raw_output"] = output
            
            resultados_finales.append(trace_result)
            
        end_time = datetime.datetime.now()
        
        # 5. Guardar resultados en JSON con metadata de tiempo
        output_data = {
            "execution_start": start_time.isoformat(),
            "execution_end": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "total_traces_processed": len(resultados_finales),
            "results": resultados_finales
        }
        
        output_dir = os.path.join(BASE_DIR, "data/output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"classification_results_{FLOW_ID}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"\n[*] ¡Proceso completado! Tiempo total: {output_data['duration_seconds']:.2f} segundos.")
        print(f"[*] Resultados guardados en: {output_file}")
            
    except Exception as e:
        print(f"\n[!] Error crítico en el Pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
