"""
Layer 5: Rank Agent (LLM-as-a-Judge) — placeholder

This module will implement the re-ranking step:
  - Receives the hydrated trace + top-K UC candidates from Layer 4 (RAG)
  - Uses the SLM to select the best match with business-level justification
  - Provides explainable output for audit purposes

Not yet implemented. Pending Layer 4 re-indexing and evaluation of
Layer 3 output quality.
"""

import json
import re
from typing import List, Dict, Optional
from llama_cpp import Llama

class RankAgent:
    """
    Layer 5: Re-ranking agent (LLM-as-a-Judge).

    Takes the top-K UC candidates from the vector search and uses
    an SLM to select the best match and write a business justification.
    """

    SYSTEM_PROMPT = """You are a Senior IT Auditor at NTT DATA.
Your task is to analyze a business context extracted from a Node-RED workflow trace, and select the BEST MATCH from a list of 5 potential Use Case candidates retrieved by a vector search.

Read the [BUSINESS CONTEXT] carefully, then evaluate each [CANDIDATE]. 
Select the single candidate that perfectly describes the business operation being performed.

OUTPUT FORMAT - Respond ONLY with valid JSON, no markdown, no extra text:
{"selectedUseCase": "UC-CODE", "confidence": "High|Medium|Low", "justification": "Detailed explanation of why this use case is the best match and why the others were rejected."}"""

    def __init__(self, llm_instance: Llama):
        """
        Initialize the RankAgent via Dependency Injection.
        Args:
            llm_instance: A pre-loaded Llama model instance.
        """
        self.llm = llm_instance

    def _build_user_prompt(self, business_context: str, candidates: List[Dict]) -> str:
        prompt = f"[BUSINESS CONTEXT]\n{business_context}\n\n[CANDIDATES]\n"
        for i, c in enumerate(candidates):
            prompt += f"Candidate {i+1}:\n"
            prompt += f" - Code: {c.get('useCaseCode', '')}\n"
            prompt += f" - Name: {c.get('displayName', '')}\n"
            prompt += f" - Description: {c.get('description', '')}\n\n"
        prompt += "[INSTRUCTION]\nEvaluate the candidates and output the JSON."
        return prompt

    def rank(self, business_context: str, candidates: List[Dict]) -> Dict:
        """
        Re-rank the top-K UC candidates and return the best match
        with a business-level justification.
        """
        if not candidates:
            return {"selectedUseCase": None, "confidence": "Low", "justification": "No candidates provided."}

        # Fast path if only 1 candidate
        if len(candidates) == 1:
            return {
                "selectedUseCase": candidates[0].get('useCaseCode'),
                "confidence": "High",
                "justification": "Only one candidate retrieved, automatically selected."
            }

        user_prompt = self._build_user_prompt(business_context, candidates)
        
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=250,
            )
            raw_output = response["choices"][0]["message"]["content"]
            
            # Defensive parsing
            clean_json = re.sub(r"```json|```", "", raw_output).strip()
            
            # Simple JSON repair for truncated ends
            if not clean_json.endswith("}"):
                if '"justification": "' in clean_json and not clean_json.rstrip().endswith('"'):
                    clean_json += '"}'
                else:
                    clean_json += "}"
            
            match = re.search(r'\{.*\}', clean_json, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                # Validate keys
                if "selectedUseCase" in data:
                    return data
            
            return {
                "selectedUseCase": candidates[0].get('useCaseCode'),
                "confidence": "Low",
                "justification": f"Fallback to top candidate due to parse error. Raw SLM output: {raw_output[:100]}"
            }
            
        except Exception as e:
            return {
                "selectedUseCase": candidates[0].get('useCaseCode'),
                "confidence": "Low",
                "justification": f"Agent error during ranking: {str(e)}"
            }
