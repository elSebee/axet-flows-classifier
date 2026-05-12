"""
Layer 2: Deterministic Keyword Router

Attempts to classify traces BEFORE invoking the SLM by analyzing
node types, configurations, and property values against keyword
fingerprints extracted from the Use Case catalog.

Three possible outcomes:
  - MATCHED:      Confident UC match found → skip SLM entirely
  - FILTERED:     Clearly not business logic → skip SLM
  - PASS_THROUGH: Ambiguous → forward to SLM (Layer 3)
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict


class RouteDecision(Enum):
    MATCHED = "matched"
    FILTERED = "filtered"
    PASS_THROUGH = "pass"


@dataclass
class RouteResult:
    decision: RouteDecision
    use_case: Optional[Dict] = None
    filter_reason: Optional[str] = None
    confidence: float = 0.0
    matched_signals: List[str] = field(default_factory=list)


class KeywordRouter:
    """
    Analyzes trace structure to find deterministic matches or filter
    non-business traces without consuming SLM compute.
    """

    # Node types that are definitively infrastructure (never business logic)
    INFRASTRUCTURE_TYPES = frozenset({
        'tab', 'comment', 'inject', 'debug', 'subflow',
        'axetflows-scheme-color', 'coding-config',
        'axetflows-db-flush',
    })

    # Node types that are "neutral" — they appear in both business and infra traces
    NEUTRAL_TYPES = frozenset({
        'function', 'link in', 'link out', 'switch', 'delay',
    })

    # Minimum overlap fraction for a confident UC match
    UC_MATCH_THRESHOLD = 0.45

    # Minimum absolute keywords that must match
    MIN_KEYWORDS_MATCHED = 2

    def __init__(self, use_cases_path: str):
        self.use_cases = self._load_catalog(use_cases_path)
        self.uc_fingerprints = self._build_fingerprints()

    def _load_catalog(self, path: str) -> List[Dict]:
        """Flatten the hierarchical catalog into a list of enabled UCs."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        flat = []
        for area in data:
            area_name = area.get('displayName', '')
            for uc in area.get('useCases', []):
                if not uc.get('enabled', True):
                    continue
                uc_flat = dict(uc)
                uc_flat['area'] = area_name
                flat.append(uc_flat)
        return flat

    # Words too generic to be useful signals on their own
    STOPWORDS = frozenset({
        'node-red', 'the', 'and', 'for', 'from', 'with', 'this', 'that',
        'are', 'can', 'has', 'into', 'using', 'based', 'ensures',
    })

    def _build_fingerprints(self) -> List[Dict]:
        """
        For each UC, extract two sets of keywords:
        - technical: hyphenated terms (e.g., 'axetflows-db-persist') → strong signals
        - generic:   single words from displayName/description → weak signals
        
        A match requires at least 1 technical keyword hit to avoid
        false positives from coincidental generic word overlap.
        """
        fingerprints = []
        for uc in self.use_cases:
            desc = uc.get('description', '').lower()
            display = uc.get('displayName', '').lower()

            technical = set()
            generic = set()

            # 1. Explicit "Technical triggers:" section
            triggers_match = re.search(
                r'technical triggers?:\s*(.+?)\.?\s*$', desc, re.IGNORECASE
            )
            if triggers_match:
                for kw in re.split(r'[,;]+', triggers_match.group(1)):
                    kw = kw.strip()
                    if len(kw) > 2 and kw not in self.STOPWORDS:
                        if '-' in kw:
                            technical.add(kw)
                        else:
                            generic.add(kw)

            # 2. Hyphenated technical terms from full description
            for term in re.findall(r'\b[a-z]+-[a-z]+(?:-[a-z]+)*\b', desc):
                if len(term) > 4 and term not in self.STOPWORDS:
                    technical.add(term)

            # 3. displayName words as generic signals
            for word in re.findall(r'\b[a-z]{3,}\b', display):
                if word not in self.STOPWORDS:
                    generic.add(word)

            fingerprints.append({
                'uc': uc,
                'technical': technical,
                'generic': generic,
                'all_keywords': technical | generic,
            })

        return fingerprints

    # ------------------------------------------------------------------
    # Signal Extraction
    # ------------------------------------------------------------------

    def _extract_trace_signals(self, trace: dict) -> Dict:
        """
        Extract structured signals from a trace dict.
        Returns node_types, node_names, prop_values, and a flat token set.
        """
        nodos = trace.get('Nodos', [])

        node_types = []
        node_names = []
        prop_values = {}
        all_tokens = set()

        for nodo_dict in nodos:
            nodo = dict(nodo_dict)
            props = nodo.pop('props', {}) or {}
            nodo.pop('linkContext', None)
            node_type = nodo.pop('_type', None)

            # Capture the node type
            if node_type:
                node_types.append(node_type)
                all_tokens.add(node_type)
                for part in node_type.split('-'):
                    if len(part) > 2:
                        all_tokens.add(part)

            # Capture node name and id
            for node_name, node_id in nodo.items():
                if not isinstance(node_id, str):
                    continue
                name_clean = node_name.strip('[]')
                if not node_name.startswith('['):
                    node_names.append(name_clean)
                for word in re.findall(r'\b[a-z]{3,}\b', name_clean.lower()):
                    all_tokens.add(word)

            # Capture prop values
            for pk, pv in props.items():
                pv_str = str(pv).lower()
                prop_values[pk] = pv_str
                all_tokens.add(pk)
                for word in re.findall(r'\b[a-z]{3,}\b', pv_str):
                    all_tokens.add(word)
                for term in re.findall(r'\b[a-z]+-[a-z]+(?:-[a-z]+)*\b', pv_str):
                    all_tokens.add(term)

        return {
            'node_types': node_types,
            'node_names': node_names,
            'prop_values': prop_values,
            'all_tokens': all_tokens,
        }

    # ------------------------------------------------------------------
    # Deterministic Filters
    # ------------------------------------------------------------------

    def _check_infrastructure(self, signals: Dict) -> Optional[str]:
        """Returns a reason string if the trace is clearly infrastructure."""
        types = signals['node_types']
        if not types:
            return None

        # Separate meaningful types from neutral glue
        meaningful = [t for t in types if t not in self.NEUTRAL_TYPES]

        if not meaningful:
            return "Only link/function nodes (pipeline glue)"

        infra_count = sum(1 for t in meaningful if t in self.INFRASTRUCTURE_TYPES)
        if infra_count == len(meaningful):
            sample = ', '.join(sorted(set(meaningful))[:3])
            return f"All nodes are infrastructure ({sample})"

        return None

    def _check_error_handling(self, signals: Dict) -> Optional[str]:
        """Returns a reason string if the trace is clearly error handling."""
        types = signals['node_types']
        if types and types[0] == 'catch':
            return "Trace starts with catch node (error handler)"
        return None

    # ------------------------------------------------------------------
    # UC Keyword Matching
    # ------------------------------------------------------------------

    def _find_best_uc_match(self, signals: Dict) -> Optional[RouteResult]:
        """
        Score all UCs against the trace signals.
        Requires at least 1 technical keyword match to prevent false positives.
        """
        trace_tokens = signals['all_tokens']

        best_score = 0.0
        best_match = None
        best_matched_kws = []
        best_fp = None

        for fp in self.uc_fingerprints:
            all_kw = fp['all_keywords']
            tech_kw = fp['technical']
            if not all_kw:
                continue

            matched = all_kw & trace_tokens
            tech_matched = tech_kw & trace_tokens

            # GATE: require at least 1 technical keyword match
            if not tech_matched:
                continue

            if len(matched) < self.MIN_KEYWORDS_MATCHED:
                continue

            score = len(matched) / len(all_kw)
            if score > best_score:
                best_score = score
                best_match = fp['uc']
                best_matched_kws = sorted(matched)
                best_fp = fp

        if best_match and best_score >= self.UC_MATCH_THRESHOLD:
            return RouteResult(
                decision=RouteDecision.MATCHED,
                use_case={
                    'useCaseCode': best_match.get('useCaseCode', ''),
                    'displayName': best_match.get('displayName', ''),
                    'area': best_match.get('area', ''),
                    'description': best_match.get('description', ''),
                    'score': round(best_score, 3),
                },
                confidence=best_score,
                matched_signals=best_matched_kws,
            )
        return None

    # ------------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------------

    def route(self, trace: dict) -> RouteResult:
        """
        Analyze a trace and decide how to handle it.

        Returns RouteResult with:
          MATCHED      → deterministic UC match, skip SLM
          FILTERED     → not business logic, skip SLM
          PASS_THROUGH → ambiguous, forward to SLM (Layer 3)
        """
        signals = self._extract_trace_signals(trace)

        # 1. Infrastructure filter
        infra_reason = self._check_infrastructure(signals)
        if infra_reason:
            return RouteResult(
                decision=RouteDecision.FILTERED,
                filter_reason=f"TechnicalInfrastructure: {infra_reason}",
                confidence=1.0,
            )

        # 2. Error handling filter
        error_reason = self._check_error_handling(signals)
        if error_reason:
            return RouteResult(
                decision=RouteDecision.FILTERED,
                filter_reason=f"ErrorHandling: {error_reason}",
                confidence=0.9,
            )

        # 3. Try UC keyword match
        uc_result = self._find_best_uc_match(signals)
        if uc_result:
            return uc_result

        # 4. No confident decision → pass to SLM
        return RouteResult(decision=RouteDecision.PASS_THROUGH)
