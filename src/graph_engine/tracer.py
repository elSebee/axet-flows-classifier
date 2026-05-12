import json
import os
from collections import defaultdict
from typing import List, Dict, Set

class FlowGraphTracer:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.nodes = {}
        self.adj = defaultdict(list)
        self.in_degree = defaultdict(int)
        
    def load_graph(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extraer nodos dependiendo de la estructura del JSON
        if 'flowsData' in data and 'flows' in data['flowsData']:
            raw_nodes = data['flowsData']['flows']
        elif isinstance(data, list):
            raw_nodes = data
        else:
            raise ValueError("Estructura de JSON no reconocida.")
            
        for node in raw_nodes:
            nid = node.get('id')
            if not nid: continue
            
            # Guardamos info relevante para el prompt
            raw_name = node.get('name') or node.get('label') or ''
            # Extraer propiedades interesantes para el SLM
            interesting_keys = ['url', 'method', 'command', 'query', 'topic', 'func', 'rules', 'property', 'action', 'table', 'database']
            props = {k: str(node[k])[:150] for k in interesting_keys if k in node and node[k]}
            
            self.nodes[nid] = {
                'id': nid,
                'type': node.get('type', 'unknown'),
                'name': str(raw_name).strip() if raw_name else '',
                'props': props
            }
            
            if nid not in self.in_degree:
                self.in_degree[nid] = 0
                
            wires = node.get('wires', [])
            # Wires en Node-RED es un array de arrays (múltiples puertos de salida)
            for port_wires in wires:
                if not port_wires: continue
                for target_id in port_wires:
                    self.adj[nid].append(target_id)
                    self.in_degree[target_id] += 1

    def find_start_and_end_nodes(self):
        """
        Identifica los nodos de inicio (sin entradas) y fin (sin salidas)
        """
        start_nodes = [nid for nid, deg in self.in_degree.items() if deg == 0 and nid in self.nodes]
        end_nodes = [nid for nid in self.nodes if len(self.adj[nid]) == 0]
        return start_nodes, end_nodes

    def extract_all_paths(self) -> List[List[str]]:
        """
        Extrae TODOS los caminos posibles desde cada nodo de inicio usando DFS.
        Evita ciclos infinitos llevando registro de los nodos visitados en la rama actual.
        """
        start_nodes, _ = self.find_start_and_end_nodes()
        all_paths = []
        
        def dfs(current_node: str, current_path: List[str], visited: Set[str]):
            # Límite de seguridad para grafos muy complejos
            if len(all_paths) >= 10000:
                return
                
            current_path.append(current_node)
            
            # Si el nodo no tiene salidas, terminamos este camino
            if len(self.adj[current_node]) == 0:
                all_paths.append(list(current_path))
            else:
                for neighbor in self.adj[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dfs(neighbor, current_path, visited)
                        visited.remove(neighbor)
                    else:
                        # Se detectó un ciclo, terminamos el camino aquí para evitar loop infinito
                        all_paths.append(list(current_path))
                        
            current_path.pop()

        for start_id in start_nodes:
            dfs(start_id, [], {start_id})
            
        return all_paths
        
    def generate_path_report(self):
        self.load_graph()
        start_nodes, end_nodes = self.find_start_and_end_nodes()
        
        print(f"[*] Total de nodos procesados: {len(self.nodes)}")
        print(f"[*] Nodos de Inicio (In-Degree 0): {len(start_nodes)}")
        print(f"[*] Nodos de Fin (Out-Degree 0): {len(end_nodes)}")
        
        print("\n[*] Extrayendo todos los caminos posibles (DFS)...")
        paths = self.extract_all_paths()
        
        print(f"[*] Se encontraron {len(paths)} trazas (caminos) únicos en la estructura estática.")
        
        # Mostrar una muestra del primer camino
        if paths:
            print("\n=== Ejemplo del Camino Estático #1 ===")
            p = paths[0]
            for i, nid in enumerate(p):
                info = self.nodes[nid]
                name_str = f"\"{info['name']}\"" if info['name'] else "Sin Nombre"
                print(f" {i+1}. [{info['type']}] {name_str} (ID: {nid})")

    def export_paths_to_json(self, output_path: str):
        paths = self.extract_all_paths()
        
        # Filtramos caminos inválidos o triviales (ej. un nodo tab solitario)
        real_paths = [p for p in paths if len(p) >= 2 and self.nodes[p[0]]['type'] != 'tab']
        
        caminos_json = []
        for i, path in enumerate(real_paths):
            nodos_list = []
            for nid in path:
                info = self.nodes[nid]
                nombre = info['name'] if info['name'] else f"[{info['type']}]"
                nodo_dict = {nombre: nid}
                if info.get('props'):
                    nodo_dict['props'] = info['props']
                nodos_list.append(nodo_dict)
                
            caminos_json.append({
                "id": f"trace_{i+1:04d}",
                "Nodos": nodos_list
            })
            
        output_data = {"Caminos": caminos_json}
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"\n[*] ¡Éxito! Se exportaron {len(real_paths)} trazas estáticas a:")
        print(f"[*] {output_path}")

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    FLOW_ID = os.getenv("FLOW_ID", "11014")
    json_path = os.path.join(base_dir, os.getenv("FLOW_JSON_DIR", "data/raw/flows_data"), f"data_{FLOW_ID}.json")
    
    tracer = FlowGraphTracer(json_path)
    tracer.generate_path_report()
    
    # Exportar el JSON a processed/static_traces/
    output_json = os.path.join(base_dir, os.getenv("STATIC_TRACES_DIR", "data/processed/static_traces"), f"traces_{FLOW_ID}.json")
    tracer.export_paths_to_json(output_json)
