"""
AXET.Flows Cognitive Classifier - Main Pipeline

Funnel Architecture:
  Layer 1: Ingestion & Hydration    → Enrich trace with metadata + logs
  Layer 2: Deterministic Routing    → Fast keyword/regex filter (skip SLM)
  Layer 3: SLM Triage               → Cognitive classification (BusinessLogic / Infra / Error)
  Layer 4: RAG Retrieval             → Semantic search against UC catalog (Top-K)
  Layer 5: LLM-as-a-Judge           → Re-ranking with explanation (future)
  Layer 6: Reporting                 → Export structured results
"""

import json
import os
import sys
import datetime

# Root del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from src.data_ingestion.tracer import FlowGraphTracer
from src.data_ingestion.hydrator import TraceHydrator
from src.keyword_router.router import KeywordRouter, RouteDecision
from src.slm_inference.slm_manager import SLMManager
from src.slm_inference.triage_agent import TriageAgent
from src.slm_inference.rank_agent import RankAgent
from src.vector_search.vector_db import VectorDatabase


def main():
    print("=" * 60)
    print("  aXet.flows Cognitive Classifier - Pipeline V2")
    print("=" * 60)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    FLOW_ID = os.getenv("FLOW_ID", "11014")

    # ---- Paths ----
    FLOW_JSON_PATH = os.path.join(BASE_DIR, os.getenv("FLOW_JSON_DIR", "data/raw/flows_data"), f"data_{FLOW_ID}.json")
    ELK_LOGS_PATH = os.path.join(BASE_DIR, os.getenv("ELK_LOGS_DIR", "data/raw/flows_logs"), f"logs_{FLOW_ID}.jsonl")
    STATIC_TRACES_PATH = os.path.join(BASE_DIR, os.getenv("STATIC_TRACES_DIR", "data/processed/static_traces"), f"traces_{FLOW_ID}.json")
    USECASES_PATH = os.path.join(BASE_DIR, os.getenv("USE_CASES_FILE", "data/raw/use_cases/useCases_v2.json"))
    USECASES_DB_PATH = os.path.join(BASE_DIR, os.getenv("PROCESSED_DIR", "data/processed"), "usecases_db.pkl")
    MODEL_PATH = os.path.join(BASE_DIR, os.getenv("MODELS_DIR", "data/models"), os.getenv("SLM_MODEL_NAME", "Phi-3-mini-4k-instruct-Q4_K_M.gguf"))
    OUTPUT_DIR = os.path.join(BASE_DIR, os.getenv("OUTPUT_DIR", "data/output"))

    print(f"\n[Config] FLOW_ID: {FLOW_ID}")

    try:
        # ==============================================================
        # LAYER 1: Ingestion & Hydration
        # ==============================================================
        print("\n--- Layer 1: Ingestion & Hydration ---")

        # 1a. Generate enriched static traces (with metadata, link resolution, etc.)
        tracer = FlowGraphTracer(FLOW_JSON_PATH)
        tracer.export_paths_to_json(STATIC_TRACES_PATH, flow_json_path=FLOW_JSON_PATH)

        with open(STATIC_TRACES_PATH, 'r', encoding='utf-8') as f:
            traces_data = json.load(f)
        caminos = traces_data.get("Caminos", [])
        flow_metadata = traces_data.get("flowMetadata", {})

        print(f"  Traces extracted: {len(caminos)}")
        if flow_metadata.get('flowName'):
            print(f"  Flow: '{flow_metadata['flowName']}' (Owner: {flow_metadata.get('ownerEmail', '?')})")

        # 1b. Load log hydrator
        hydrator = TraceHydrator(ELK_LOGS_PATH)

        # ==============================================================
        # LAYER 2: Deterministic Routing
        # ==============================================================
        print("\n--- Layer 2: Deterministic Routing ---")

        router = KeywordRouter(USECASES_PATH)

        # Route all traces through Layer 2
        layer2_matched = []    # Quick Win UC matches
        layer2_filtered = []   # Infrastructure / Error (no business value)
        layer2_pass = []       # Ambiguous → send to SLM (Layer 3)

        for traza in caminos:
            route_result = router.route(traza)
            if route_result.decision == RouteDecision.MATCHED:
                layer2_matched.append((traza, route_result))
            elif route_result.decision == RouteDecision.FILTERED:
                layer2_filtered.append((traza, route_result))
            else:
                layer2_pass.append(traza)

        print(f"  MATCHED (Quick Win):    {len(layer2_matched)} traces → Skip SLM")
        print(f"  FILTERED (Infra/Error): {len(layer2_filtered)} traces → Skip SLM")
        print(f"  PASS → SLM:            {len(layer2_pass)} traces → Layer 3")

        # ==============================================================
        # LAYER 3: SLM Triage (only for PASS_THROUGH traces)
        # ==============================================================
        print(f"\n--- Layer 3: SLM Triage ({len(layer2_pass)} traces) ---")

        # Pre-check: do we need the SLM at all?
        need_slm = len(layer2_pass) > 0
        triage_agent = None

        if need_slm:
            if not os.path.exists(MODEL_PATH):
                print(f"  [!] Error: SLM not found at {MODEL_PATH}")
                return

        # ==============================================================
        # LAYER 4: RAG Retrieval (Vector Search)
        # ==============================================================
        print(f"\n--- Layer 4: RAG Retrieval ---")

        if not os.path.exists(USECASES_DB_PATH):
            print(f"  [!] Error: Vector DB not found at {USECASES_DB_PATH}. Run vector_db.py first.")
            return

        print(f"[*] Cargando Vector Search...")
        vector_db = VectorDatabase()
        vector_db.load(USECASES_DB_PATH)
        print(f"  VectorDB loaded: {len(vector_db.metadata)} UCs indexed")

        print(f"[*] Configurando SLM Agents (Inyección de Dependencias)...")
        slm_manager = SLMManager(model_path=MODEL_PATH)
        triage_agent = TriageAgent(llm_instance=slm_manager.llm)
        rank_agent = RankAgent(llm_instance=slm_manager.llm)

        # ==============================================================
        # PROCESSING LOOP
        # ==============================================================
        print(f"\n--- Processing {len(caminos)} traces ---")
        start_time = datetime.datetime.now()

        resultados_finales = []

        # ---- Process Layer 2: MATCHED traces (instant, no SLM) ----
        for traza, route_result in layer2_matched:
            trace_id = traza.get('id', '?')
            uc = route_result.use_case
            resultados_finales.append({
                "trace_id": trace_id,
                "nodos": len(traza.get("Nodos", [])),
                "trace_type": "BusinessLogic",
                "reasoning": f"Deterministic match via keyword signals: {route_result.matched_signals}",
                "slm_use_case_name": uc['displayName'],
                "slm_business_context": uc.get('description', ''),
                "top_matches": [{
                    "useCaseCode": uc['useCaseCode'],
                    "displayName": uc['displayName'],
                    "area": uc.get('area', ''),
                    "score": uc.get('score', 1.0),
                }],
                "status": "Mapped (Deterministic)",
                "layer_resolved": 2,
            })

        # ---- Process Layer 2: FILTERED traces (instant, no SLM) ----
        for traza, route_result in layer2_filtered:
            trace_id = traza.get('id', '?')
            resultados_finales.append({
                "trace_id": trace_id,
                "nodos": len(traza.get("Nodos", [])),
                "trace_type": route_result.filter_reason.split(":")[0] if route_result.filter_reason else "TechnicalInfrastructure",
                "reasoning": route_result.filter_reason or "Filtered by deterministic rules",
                "slm_use_case_name": "",
                "slm_business_context": "",
                "top_matches": [],
                "status": "System Overhead (Non-billable)",
                "layer_resolved": 2,
            })

        # ---- Process Layer 2: PASS_THROUGH traces (SLM + Vector Search) ----
        total_pass = len(layer2_pass)
        parse_errors = 0

        for idx, traza in enumerate(layer2_pass):
            trace_id = traza.get('id', '?')
            print(f"  [{idx+1}/{total_pass}] {trace_id}...", end=" ", flush=True)

            trace_result = {
                "trace_id": trace_id,
                "nodos": len(traza.get("Nodos", [])),
                "trace_type": "Unknown",
                "reasoning": "",
                "slm_use_case_name": "",
                "slm_business_context": "",
                "top_matches": [],
                "status": "Unmapped",
                "layer_resolved": 3,
            }

            # Layer 3: SLM Triage
            hydrated_data = hydrator.hydrate(traza, flow_metadata)
            triage = triage_agent.classify(hydrated_data)

            trace_result["trace_type"] = triage.trace_type
            trace_result["reasoning"] = triage.reasoning
            trace_result["slm_use_case_name"] = triage.use_case_name
            trace_result["slm_business_context"] = triage.business_context

            if not triage.parse_success:
                trace_result["status"] = "Parse Error"
                trace_result["raw_output"] = triage.raw_output
                parse_errors += 1
                print("PARSE_ERROR")
                resultados_finales.append(trace_result)
                continue

            # Layer 4: RAG Retrieval (only for BusinessLogic)
            if triage.trace_type == "BusinessLogic" and triage.business_context:
                matches = vector_db.search(triage.business_context, top_k=5)

                trace_result["top_matches"] = matches

                if matches:
                    # Layer 5: Re-ranking (SLM-as-a-Judge)
                    rank_result = rank_agent.rank(triage.business_context, matches)
                    
                    trace_result["final_use_case"] = rank_result.get("selectedUseCase")
                    trace_result["confidence"] = rank_result.get("confidence")
                    trace_result["justification"] = rank_result.get("justification")
                    trace_result["status"] = "Ranked by SLM"
                    trace_result["layer_resolved"] = 5
                    
                    print(f"OK → {rank_result.get('selectedUseCase')} ({rank_result.get('confidence')})")
                else:
                    trace_result["status"] = "Low Confidence / Requires Human Review"
                    print(f"OK (Low Confidence)")
            else:
                trace_result["status"] = "System Overhead (Non-billable)"
                print(f"OK ({triage.trace_type})")

            resultados_finales.append(trace_result)

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        # ==============================================================
        # LAYER 6: Reporting
        # ==============================================================
        print(f"\n--- Layer 6: Reporting ---")

        # Sort results by trace_id for consistency
        resultados_finales.sort(key=lambda r: r.get("trace_id", ""))

        # Compute summary stats
        status_counts = {}
        layer_counts = {}
        for r in resultados_finales:
            s = r.get("status", "Unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
            l = r.get("layer_resolved", 0)
            layer_counts[l] = layer_counts.get(l, 0) + 1

        output_data = {
            "pipeline_version": "2.0",
            "flow_id": FLOW_ID,
            "flow_name": flow_metadata.get("flowName", ""),
            "execution_start": start_time.isoformat(),
            "execution_end": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "total_traces": len(resultados_finales),
            "summary": {
                "by_status": status_counts,
                "by_layer_resolved": {f"layer_{k}": v for k, v in sorted(layer_counts.items())},
                "slm_calls": total_pass,
                "slm_parse_errors": parse_errors,
                "slm_saved_by_router": len(layer2_matched) + len(layer2_filtered),
            },
            "results": resultados_finales,
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"classification_results_{FLOW_ID}.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Total traces: {len(resultados_finales)}")
        print(f"  SLM calls: {total_pass} (saved {len(layer2_matched)+len(layer2_filtered)} via Router)")
        if parse_errors:
            print(f"  Parse errors: {parse_errors}/{total_pass} ({parse_errors/max(total_pass,1)*100:.1f}%)")
        print(f"\n  Status breakdown:")
        for status, count in sorted(status_counts.items()):
            print(f"    {status}: {count}")
        print(f"\n  Results saved to: {output_file}")

    except Exception as e:
        print(f"\n[!] Critical pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
