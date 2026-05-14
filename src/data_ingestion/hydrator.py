"""
Layer 1b: Trace Hydrator

Loads ELK runtime logs and combines them with static traces
to produce token-budget-aware prompts ready for the SLM.

Responsibilities:
  - Index log events by node_id for O(1) lookup
  - Translate technical Node-RED props into readable business labels
  - Apply 3-level compression to keep prompts within the SLM token budget:
      Level 1 (always): Skip JS code (func), clean node names
      Level 2 (if needed): Reduce log samples to 1 per node
      Level 3 (if needed): Collapse middle nodes into a compact summary
"""

import json
import os


class TraceHydrator:
    """
    Combines a static trace (from tracer.py) with runtime logs
    to generate a hydrated, token-aware prompt for the SLM Triage Agent.
    """

    def __init__(self, logs_path: str):
        """
        Args:
            logs_path: Path to the JSONL file containing ELK runtime events.
        """
        self.logs_path = logs_path
        self.events_by_node = self._load_events()

    def _load_events(self) -> dict:
        """
        Read the JSONL log file and group messages by node_id.
        Enables O(1) lookup during hydration.
        """
        events = {}
        if not os.path.exists(self.logs_path):
            print(f"[!] Aviso: No se encontraron logs en {self.logs_path}")
            return events

        with open(self.logs_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    source = json.loads(line)

                    # Extract node_id
                    node_info = source.get("node") or source.get("sourceNode")
                    event_info = source.get("event", {})
                    node_id = None

                    if isinstance(node_info, dict):
                        node_id = node_info.get("id")
                    elif isinstance(node_info, str):
                        node_id = node_info

                    if not node_id and isinstance(event_info, dict):
                        node_id = event_info.get("nodeid")

                    # Extract message or payload
                    msg = source.get("msg", "")
                    if not msg:
                        if "payload" in source:
                            msg = str(source["payload"])
                        elif isinstance(event_info, dict) and "value" in event_info:
                            msg = event_info.get("value")

                    if node_id and msg and "Connect Timeout Error" not in str(msg):
                        clean_msg = str(msg).strip()[:200]
                        if node_id not in events:
                            events[node_id] = []
                        if clean_msg not in events[node_id]:  # deduplicate
                            events[node_id].append(clean_msg)

                except Exception:
                    continue

        return events

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    # Technical Node-RED properties → readable business labels for the SLM
    PROP_LABELS = {
        'url': 'API Endpoint',
        'method': 'HTTP Method',
        'command': 'System Command',
        'query': 'Database Query',
        'action': 'UI Action',
        'property': 'Data Binding',
        'dbName': 'Database/Collection',
        'message': 'UI Message',
        'associatedEntityMetamodel': 'Business Entity',
        'associatedEntityFormType': 'Form Type',
        'bindingProperty': 'Data Source',
        'sortProperty': 'Sort Order',
        'credentialname': 'Integration Credential',
        'table': 'Database Table',
        'database': 'Database Name',
        'topic': 'Message Topic',
        'rules': 'Routing Rules',
        'paginator': 'Pagination',
    }

    # Props always excluded — JS code and internal coordinates add noise
    SKIP_PROPS = frozenset({'func', 'noerr', 'wires', 'x', 'y', 'z', 'outputs'})

    # Token budget: context_window(2560) - system_prompt(~306) - max_output(200) - overhead(50)
    TOKEN_BUDGET = 2004
    CHARS_PER_TOKEN = 4  # rough estimate for Phi-3 tokenizer

    # Nodes to keep at each end when collapsing middle (Level 3 compression)
    EDGE_NODES = 4

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_name(name: str) -> str:
        """Remove escape chars and whitespace artifacts from node names."""
        cleaned = name.replace('\\n', ' ').replace('\\t', ' ')
        cleaned = cleaned.replace('\n', ' ').replace('\t', ' ')
        return ' '.join(cleaned.split()).strip()

    def _build_step(self, i: int, nodo_dict_original: dict, max_logs: int = 3) -> str:
        """
        Build the prompt text for a single node step.

        Args:
            i: Step index (0-based).
            nodo_dict_original: Raw node dict from the static trace.
            max_logs: Max runtime log samples to include.
        """
        nodo_dict = dict(nodo_dict_original)
        props = nodo_dict.pop('props', None)
        link_context = nodo_dict.pop('linkContext', None)
        nodo_dict.pop('_type', None)

        for node_name, node_id in nodo_dict.items():
            if not isinstance(node_id, str):
                continue

            clean_name = self._clean_name(node_name)
            step_info = f"Step {i+1}: [{clean_name}] (ID: {node_id})"

            if link_context:
                step_info += f"\n  - Connected to: {link_context}"

            if props:
                labeled_props = []
                for k, v in props.items():
                    if k in self.SKIP_PROPS:
                        continue
                    label = self.PROP_LABELS.get(k, k)
                    v_str = str(v)
                    if len(v_str) > 100:
                        v_str = v_str[:100] + '...'
                    labeled_props.append(f"{label}: '{v_str}'")
                if labeled_props:
                    step_info += f"\n  - Config: {', '.join(labeled_props)}"

            log_samples = self.events_by_node.get(node_id, [])
            if log_samples:
                samples = " | ".join(log_samples[:max_logs])
                step_info += f"\n  - Runtime log: \"{samples}\""

            return step_info
        return ""

    def _build_compact_step(self, i: int, nodo_dict_original: dict) -> str:
        """One-line node summary for Level 3 middle-node compression."""
        nodo_dict = dict(nodo_dict_original)
        nodo_dict.pop('props', None)
        nodo_dict.pop('linkContext', None)
        node_type = nodo_dict.pop('_type', '?')

        for node_name, node_id in nodo_dict.items():
            if not isinstance(node_id, str):
                continue
            clean_name = self._clean_name(node_name)
            if clean_name.startswith('[') and clean_name.endswith(']'):
                return node_type
            return f"{clean_name}({node_type})"
        return node_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hydrate(self, static_trace: dict, flow_metadata: dict = None) -> str:
        """
        Combine a static trace with runtime logs into a token-budget-aware prompt.

        Compression levels applied in order until the prompt fits:
          Level 1 (always):   Skip JS func code, clean escape chars in names
          Level 2 (if needed): Reduce log samples from 3 → 1 per node
          Level 3 (if needed): Collapse middle nodes into a one-line summary

        Args:
            static_trace: A single trace dict from traces_<flow_id>.json.
            flow_metadata: Optional dict with flowName, ownerEmail, etc.

        Returns:
            Formatted string ready to be injected into the SLM user prompt.
        """
        trace_id = static_trace.get("id", "Unknown")
        nodos = static_trace.get("Nodos", [])

        # Header — always present
        header_parts = []
        if flow_metadata and flow_metadata.get('flowName'):
            header_parts.append(f"APPLICATION: {flow_metadata['flowName']}")
        header_parts.append(f"TRACE ID: {trace_id}")
        header_parts.append(f"TOTAL STEPS: {len(nodos)}")
        header_parts.append("EXECUTION PATH:")
        header = "\n".join(header_parts)

        # Level 1: full detail, no func (always applied)
        steps = [self._build_step(i, n, max_logs=3) for i, n in enumerate(nodos)]
        prompt = header + "\n" + "\n".join(steps)
        if len(prompt) // self.CHARS_PER_TOKEN <= self.TOKEN_BUDGET:
            return prompt

        # Level 2: reduce logs to 1 sample
        steps = [self._build_step(i, n, max_logs=1) for i, n in enumerate(nodos)]
        prompt = header + "\n" + "\n".join(steps)
        if len(prompt) // self.CHARS_PER_TOKEN <= self.TOKEN_BUDGET:
            return prompt

        # Level 3: collapse middle nodes
        edge = self.EDGE_NODES
        if len(nodos) > edge * 2:
            head = [self._build_step(i, n, max_logs=1) for i, n in enumerate(nodos[:edge])]
            tail = [self._build_step(i, n, max_logs=1) for i, n in enumerate(nodos[-edge:], len(nodos) - edge)]
            middle = nodos[edge:-edge]
            middle_labels = [self._build_compact_step(i, n) for i, n in enumerate(middle, edge)]
            mid_line = f"--- Steps {edge+1}-{len(nodos)-edge}: {' → '.join(middle_labels)} ---"
            prompt = header + "\n" + "\n".join(head + [mid_line] + tail)

        return prompt


# ---------------------------------------------------------------------------
# Backwards-compat alias — pipelines that imported TraceSummarizer still work
# ---------------------------------------------------------------------------
class TraceSummarizer(TraceHydrator):
    """Deprecated alias for TraceHydrator. Use TraceHydrator instead."""

    def hydrate_trace(self, static_trace: dict, flow_metadata: dict = None) -> str:
        """Deprecated: use hydrate() instead."""
        return self.hydrate(static_trace, flow_metadata)
