import json
from datetime import datetime

class GraphReconstructor:
    def __init__(self, node_red_flow_path):
        """
        Inicializa el reconstructor cargando el grafo estático (JSON de Node-RED).
        Crea un diccionario rápido para buscar nodos por su ID.
        """
        self.node_dict = {}
        self._load_flow(node_red_flow_path)

    def _load_flow(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Iterar sobre la definición de los flujos
        flows = data.get("flowsData", {}).get("flows", [])
        for node in flows:
            node_id = node.get("id")
            if node_id:
                # Guardar info relevante: tipo y etiqueta/nombre
                self.node_dict[node_id] = {
                    "type": node.get("type", "unknown"),
                    "label": node.get("label", node.get("name", "Unnamed Node"))
                }

    def process_elk_logs(self, elk_logs_path):
        """Lee los logs desde un archivo JSON y los procesa."""
        with open(elk_logs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self.process_elk_logs_from_data(data)

    def process_elk_logs_from_data(self, data):
        """
        Procesa los logs de ELK desde un diccionario, filtra el ruido, y construye la Traza Canónica.
        """
        hits = data.get("hits", {}).get("hits", [])
        
        # Agruparemos los eventos útiles en una lista secuencial
        # En un entorno real, se debería agrupar por TraceID o CorrelationID.
        # Aquí agruparemos por FLOWS_ID asumiendo un bloque continuo para propósitos de la PoC.
        
        traces_by_flow = {}

        for hit in hits:
            source = hit.get("_source", {})
            msg = source.get("msg", "")
            
            # Filtro de Ruido (Data Masking & Noise Reduction)
            if not msg or "Connect Timeout Error" in msg:
                continue
                
            flows_id = source.get("FLOWS_ID", "UNKNOWN_FLOW")
            
            if flows_id not in traces_by_flow:
                traces_by_flow[flows_id] = []
                
            # Extraer Node ID del log dinámico (puede estar en 'node' o 'sourceNode')
            node_info = source.get("node") or source.get("sourceNode")
            node_id = None
            
            if isinstance(node_info, dict):
                node_id = node_info.get("id")
            elif isinstance(node_info, str):
                node_id = node_info

            # Enriquecimiento cruzando con el Grafo Estático
            node_context = self.node_dict.get(node_id, {"type": "unknown", "label": "Unknown Node"}) if node_id else {"type": "system", "label": "System/Unknown"}
            
            # Limpiar PII y condensar msg (ejemplo sencillo)
            clean_msg = msg.strip()
            
            event = {
                "timestamp": source.get("timestamp"),
                "node_id": node_id,
                "node_type": node_context["type"],
                "node_label": node_context["label"],
                "message": clean_msg
            }
            traces_by_flow[flows_id].append(event)
            
        # Generar texto semántico
        canonical_traces = {}
        for f_id, events in traces_by_flow.items():
            # Ordenar por timestamp (ElasticSearch suele darlos ordenados, pero aseguramos)
            try:
                events.sort(key=lambda x: datetime.fromisoformat(x["timestamp"].replace("Z", "+00:00")) if x["timestamp"] else datetime.min)
            except Exception:
                pass

            canonical_text = [f"Inicio de Traza (Flow ID: {f_id})"]
            
            for idx, ev in enumerate(events, 1):
                canonical_text.append(f"{idx}. [{ev['node_type']}] ({ev['node_label']}): {ev['message']}")
                
            canonical_text.append("Fin de Traza.")
            canonical_traces[f_id] = "\n".join(canonical_text)

        return canonical_traces
