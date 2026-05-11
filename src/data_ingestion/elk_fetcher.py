import os
import json
import requests
from typing import Dict, Any, Generator

class ElkLogFetcher:
    def __init__(self, kibana_url: str, index: str, api_key: str):
        """
        Inicializa el cliente para extraer logs a través del Kibana Proxy usando ApiKey.
        """
        self.kibana_url = kibana_url.rstrip('/')
        self.index = index
        self.api_key = api_key
        
        self.headers = {
            "kbn-xsrf": "true",
            "Content-Type": "application/json",
            "Authorization": f"ApiKey {self.api_key}"
        }
        
        # Desactivar advertencias de SSL para redes corporativas
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.verify_ssl = False 

    def _es_request(self, method: str, path: str, body: dict = None) -> requests.Response:
        """Helper para hacer requests al Kibana Proxy."""
        url = f"{self.kibana_url}/api/console/proxy"
        response = requests.post(
            url,
            params={"path": path, "method": method},
            headers=self.headers,
            json=body,
            verify=self.verify_ssl
        )
        response.raise_for_status()
        return response

    def fetch_flows_logs(self, flows_id: str, start_time: str, end_time: str, batch_size: int = 5000) -> Generator[Dict[str, Any], None, None]:
        """
        Extrae logs para un FLOWS_ID usando PIT y Search After.
        """
        # 1. Crear un PIT (Point in Time)
        print("[*] Abriendo Point In Time (PIT)...")
        pit_resp = self._es_request("POST", f"/{self.index}/_pit?keep_alive=5m")
        pit_id = pit_resp.json()["id"]

        query = {
            "size": batch_size,
            "query": {
                "bool": {
                    "must": [
                        { "match": { "FLOWS_ID": flows_id } },
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time,
                                    "lt": end_time
                                }
                            }
                        }
                    ]
                }
            },
            "pit": {
                "id": pit_id,
                "keep_alive": "5m"
            },
            "sort": [
                { "timestamp": "asc" }
            ]
        }

        try:
            search_after = None
            total_fetched = 0
            
            while True:
                if search_after:
                    query["search_after"] = search_after
                
                resp = self._es_request("GET", "/_search", body=query)
                data = resp.json()
                
                hits = data.get('hits', {}).get('hits', [])
                if not hits:
                    break
                    
                for hit in hits:
                    yield hit['_source']
                    
                total_fetched += len(hits)
                search_after = hits[-1].get("sort")
                
                # Actualizar el PIT en la query
                query["pit"]["id"] = data.get("pit_id", pit_id)
                
        finally:
            # Eliminar el PIT al terminar
            print("[*] Cerrando Point In Time (PIT)...")
            try:
                self._es_request("DELETE", "/_pit", body={"id": query["pit"]["id"]})
            except Exception as e:
                print(f"[!] Aviso: No se pudo cerrar el PIT limpiamente ({e})")

    def download_to_file(self, flows_id: str, start_time: str, end_time: str, output_path: str):
        """Descarga masiva de logs a archivo JSONL."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"[*] Iniciando descarga de logs a través de Kibana Proxy...")
        print(f"[*] FLOWS_ID: {flows_id} | Rango: {start_time} -> {end_time}")
        
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for log_source in self.fetch_flows_logs(flows_id, start_time, end_time):
                f.write(json.dumps(log_source) + "\n")
                count += 1
                if count % 10000 == 0:
                    print(f"   ... {count} logs descargados")
                    
        print(f"[*] ¡Descarga completada! {count} logs guardados en {output_path}")

if __name__ == "__main__":
    # CONFIGURACIÓN KIBANA PROXY
    KIBANA_URL = "https://kibana-prod.deptapps-instances.automation-coe.com"
    ELK_INDEX = "deptapps-audit-index-logs"
    API_KEY = "Nk83ajFaMEI2b1VXSTZvOVNHUmk6V3FSLW4tOG9UXy1jdEF0YnRPUXpnZw=="
    
    # PARÁMETROS DEL REQUERIMIENTO
    #TARGET_FLOWS = ["3834", "9363", "5458", "10647"]
    TARGET_FLOWS = ["11014"]
    START_TIME = "2025-04-23T00:00:00"
    END_TIME = "2025-04-24T00:00:00"
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fetcher = ElkLogFetcher(KIBANA_URL, ELK_INDEX, API_KEY)
    
    for flow_id in TARGET_FLOWS:
        save_path = os.path.join(base_dir, f"data/raw/flowsLogs/logs_{flow_id}.jsonl")
        if os.path.exists(save_path):
            print(f"[*] El archivo para FLOWS_ID {flow_id} ya existe. Omitiendo...")
            continue
            
        print(f"\n{'='*50}\n[*] Procesando FLOWS_ID: {flow_id}\n{'='*50}")
        fetcher.download_to_file(flow_id, START_TIME, END_TIME, save_path)
