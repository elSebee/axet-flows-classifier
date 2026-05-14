"""
Layer 3: SLM Triage Agent

Responsible for the cognitive classification of traces that passed
through Layer 2's deterministic filter (PASS_THROUGH traces only).

The agent uses a local SLM (Phi-3) to:
  1. Classify the trace type (BusinessLogic / TechnicalInfrastructure / ErrorHandling)
  2. Generate a short business-oriented name for the use case
  3. Produce a 1-2 sentence "business context" description suitable
     for semantic search against the UC catalog (Layer 4)

This module owns the SLM instance lifecycle (lazy-loaded, persistent)
and the prompt engineering. It does NOT perform vector search or
UC catalog matching — that belongs to Layer 4.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from llama_cpp import Llama


@dataclass
class TriageResult:
    """Output of Layer 3: what the SLM determined about a trace."""
    trace_type: str = "Unknown"          # BusinessLogic | TechnicalInfrastructure | ErrorHandling
    reasoning: str = ""                  # Step-by-step SLM reasoning
    use_case_name: str = ""              # Short name (max 5 words)
    business_context: str = ""           # 1-2 sentence description for vector search
    raw_output: str = ""                 # Raw SLM output for debugging
    parse_success: bool = False          # Whether JSON parsing succeeded


class TriageAgent:
    """
    Encapsulates SLM-based trace classification.
    
    Design principles:
      - Single Responsibility: only does triage, no vector search
      - Lazy Loading: SLM is loaded on first inference, reused after
      - Defensive Parsing: multiple strategies to extract JSON from SLM output
    """

    # ---- SLM Configuration ----
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 200   # Increased from 150 to reduce truncated JSON

    def __init__(self, llm_instance: Llama):
        """
        Initialize the TriageAgent via Dependency Injection.
        Args:
            llm_instance: A pre-loaded Llama model instance.
        """
        self.llm = llm_instance

    # ------------------------------------------------------------------
    # Prompt Engineering
    # ------------------------------------------------------------------

    SYSTEM_PROMPT = """You are a Senior Business Analyst at NTT DATA analyzing Node-RED workflow execution traces.

TASK: Analyze the trace and classify it. Each trace is a sequence of nodes with their configurations and runtime logs.

CLASSIFICATION RULES:
- "BusinessLogic": The trace performs a meaningful business operation such as:
  * Processing form submissions or user input
  * Reading/writing database records
  * Chatbot or AI interactions (extract-answer, RAG)
  * Document processing or file operations
  * User authentication or session management
  * View rendering with data binding to business entities

- "TechnicalInfrastructure": The trace performs internal setup or utility work:
  * Initializing default values or global settings
  * Cache clearing or database flushing
  * Debug logging or monitoring
  * Pure link-routing without business operations

- "ErrorHandling": The trace catches or processes errors:
  * Error catch and notification flows
  * Validation failure responses

OUTPUT FORMAT - Respond ONLY with valid JSON, no markdown, no extra text:
{"traceType": "BusinessLogic", "reasoning": "brief explanation", "useCaseName": "Max Five Words", "businessContext": "1-2 sentence description of the business purpose."}"""

    def _build_user_prompt(self, hydrated_data: str) -> str:
        """Build the user prompt from hydrated trace data."""
        return f"""[TRACE DATA]
{hydrated_data}

[INSTRUCTION]
Classify this trace. Return ONLY the JSON object."""

    # ------------------------------------------------------------------
    # Inference & Parsing
    # ------------------------------------------------------------------

    def _run_inference(self, system_prompt: str, user_prompt: str) -> str:
        """Execute SLM inference and return raw text output."""
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.DEFAULT_TEMPERATURE,
            max_tokens=self.DEFAULT_MAX_TOKENS,
        )
        return response["choices"][0]["message"]["content"]

    def _parse_slm_output(self, raw: str) -> dict:
        """
        Defensive JSON parsing with multiple fallback strategies.
        Handles common SLM output issues: markdown wrapping, trailing text,
        incomplete JSON, etc.
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON block from markdown
        clean = re.sub(r"```json\s*|```\s*", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Find the first { ... } block
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Try to repair truncated JSON (missing closing brace/quote)
        candidate = raw.strip()
        if candidate.startswith('{') and not candidate.endswith('}'):
            # Count open/close braces
            open_braces = candidate.count('{') - candidate.count('}')
            # Try adding missing closing braces and quote
            for repair in [
                candidate + '"}',
                candidate + '"' + '}' * open_braces,
                candidate + '}' * open_braces,
            ]:
                try:
                    return json.loads(repair)
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not parse SLM output as JSON: {raw[:200]}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, hydrated_data: str) -> TriageResult:
        """
        Main entry point: classify a hydrated trace.
        
        Args:
            hydrated_data: Pre-formatted trace string from TraceSummarizer.hydrate_trace()
        
        Returns:
            TriageResult with classification, reasoning, and business context.
        """
        user_prompt = self._build_user_prompt(hydrated_data)
        raw_output = self._run_inference(self.SYSTEM_PROMPT, user_prompt)

        result = TriageResult(raw_output=raw_output)

        try:
            data = self._parse_slm_output(raw_output)
            result.trace_type = data.get("traceType", "Unknown")
            result.reasoning = data.get("reasoning", "")
            result.use_case_name = data.get("useCaseName", "")
            result.business_context = data.get("businessContext", "")
            result.parse_success = True
        except (ValueError, KeyError) as e:
            result.trace_type = "ParseError"
            result.reasoning = str(e)

        return result
