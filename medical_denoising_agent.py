"""
PCRAgent: Framework for Medical Dialogue Denoising
Detector-Editor-Arbiter Pipeline

LLM prompts are maintained centrally in pcr_agent_prompt.py (PCRAgentPrompts).

Framework Overview:
1. Detector: Identifies potential edit signals with standardized tags
2. Editor: Processes deterministic and candidate edits with scoring
3. Arbiter: LLM-based final decision making and conflict resolution
4. Evaluation: Multi-dimensional quality assessment

Output format for edits:
[{
  "start_char": int,
  "end_char": int,
  "op": "REPLACE" | "DELETE" | "INSERT",
  "cand_texts": ["candidate1", "candidate2", ...],
  "score": float,
  "tag": "RPT|SPL|GRM|AMB|NOS",
  "edit_type": "deterministic|candidate"
}, ...]
"""

from typing import List, Dict, Tuple, Optional, Callable
import os
import re, json, itertools
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from pcr_agent_prompt import PCRAgentPrompts  # pyright: ignore[reportMissingImports]

from llm_api_utils import (
    chat_completions_create,
    create_openai_client,
    estimate_max_tokens,
    print_api_error,
)
from dataclasses import dataclass, asdict
from collections import Counter
import time
from functools import wraps

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import symspellpy
from symspellpy import SymSpell, Verbosity
try:
    from openai import APIConnectionError, APIError, RateLimitError
except ImportError:
    # Backward compatibility for older SDK versions
    APIConnectionError = Exception
    APIError = Exception
    RateLimitError = Exception

# ==================== API retry utilities ====================
def retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2):
    """
    Retry decorator for API calls.

    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Exponential backoff factor
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (APIConnectionError, RateLimitError) as e:
                    last_exception = e
                    print_api_error(
                        e,
                        context=getattr(func, "__name__", "API call"),
                        attempt=attempt + 1,
                        max_attempts=max_retries,
                    )
                    if attempt < max_retries - 1:
                        delay = initial_delay * (backoff_factor ** attempt)
                        time.sleep(delay)
                        continue
                    raise
                except APIError as e:
                    print_api_error(
                        e,
                        context=getattr(func, "__name__", "API call"),
                        show_traceback=True,
                    )
                    raise
                except Exception as e:
                    if "Connection" in str(e) or "timeout" in str(e).lower():
                        last_exception = e
                        print_api_error(
                            e,
                            context=getattr(func, "__name__", "API call"),
                            attempt=attempt + 1,
                            max_attempts=max_retries,
                        )
                        if attempt < max_retries - 1:
                            delay = initial_delay * (backoff_factor ** attempt)
                            time.sleep(delay)
                            continue
                    print_api_error(
                        e,
                        context=getattr(func, "__name__", "API call"),
                        show_traceback=True,
                    )
                    raise

            if last_exception:
                raise last_exception
        return wrapper
    return decorator
from symspellpy import SymSpell, Verbosity
import nltk
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    print(f"Warning: NLTK punkt download failed: {e}")
    print("This does not affect core functionality; you may continue.")

# ---------- Helper: normalize edits ----------
def normalize_edits(edits):
    """
    Accepts a list of SpanEdit or dict-like edits and returns a list of dicts with keys:
    'start_char', 'end_char', 'op', 'cand_texts', 'score'.
    """
    norm = []
    for e in edits:
        if isinstance(e, SpanEdit):
            d = e.to_dict()
        elif isinstance(e, dict):
            d = e.copy()
        else:
            try:
                d = {
                    "start_char": getattr(e, "start_char"),
                    "end_char": getattr(e, "end_char"),
                    "op": getattr(e, "op"),
                    "cand_texts": getattr(e, "cand_texts"),
                    "score": getattr(e, "score", 0.0)
                }
            except Exception:
                continue

        if "start" in d and "start_char" not in d:
            d["start_char"] = d.pop("start")
        if "end" in d and "end_char" not in d:
            d["end_char"] = d.pop("end")

        d.setdefault("start_char", 0)
        d.setdefault("end_char", d["start_char"])
        d.setdefault("op", "REPLACE")
        d.setdefault("cand_texts", [""])
        d.setdefault("score", 0.0)

        norm.append(d)
    return norm


# --------------------------
# Data structure
# --------------------------
@dataclass
class SpanEdit:
    start_char: int
    end_char: int
    op: str  # "REPLACE"/"DELETE"/"INSERT"
    cand_texts: List[str]
    score: float
    tag: str = ""  # "RPT|SPL|GRM|AMB|NOS"
    edit_type: str = "deterministic"  # "deterministic|candidate"
    detector_name: str = ""  # Source detector name

    def to_dict(self):
        return asdict(self)

class BaseExtractorModule:
    def detect(self, text: str) -> List[SpanEdit]:
        raise NotImplementedError()

# --------------------------
# 1) GEC / seq2edit
# --------------------------
import re
import difflib
from typing import List

class GECTagger(BaseExtractorModule):
    def __init__(self, model_name_or_path=r"C:\Users\Laptop4\Desktop\models\grammar_error_correcter_v1"):
        # Load tokenizer and model from local path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    def detect(self, text: str) -> List[SpanEdit]:
        # Long text: raise sequence limit; very long dialogues should rely on LLM detectors
        max_len = min(1024, max(256, len(text) // 2 + 64))
        inputs = self.tokenizer([text], max_length=max_len, truncation=True, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_len)
        corrected = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        edits = []
        if corrected != text:
            orig_tokens = text.split()
            corr_tokens = corrected.split()

            # Align with difflib
            diff = difflib.SequenceMatcher(None, orig_tokens, corr_tokens)

            for tag, i1, i2, j1, j2 in diff.get_opcodes():
                if tag == "replace":
                    span = re.search(re.escape(" ".join(orig_tokens[i1:i2])), text)
                    if span:
                        edits.append(SpanEdit(
                            span.start(),
                            span.end(),
                            "REPLACE",
                            [" ".join(corr_tokens[j1:j2])],
                            0.9,
                            tag="GRM",
                            edit_type="deterministic",
                            detector_name="GEC"
                        ))
                elif tag == "insert":
                    # Insertion point: end of original token span
                    pos = (re.search(re.escape(" ".join(orig_tokens[:i1])), text).end()
                           if i1 > 0 else 0)
                    edits.append(SpanEdit(
                        pos,
                        pos,
                        "INSERT",
                        [" ".join(corr_tokens[j1:j2])],
                        0.8,
                        tag="GRM",
                        edit_type="deterministic",
                        detector_name="GEC"
                    ))
                elif tag == "delete":
                    span = re.search(re.escape(" ".join(orig_tokens[i1:i2])), text)
                    if span:
                        edits.append(SpanEdit(
                            span.start(),
                            span.end(),
                            "DELETE",
                            [""],
                            0.8,
                            tag="GRM",
                            edit_type="deterministic",
                            detector_name="GEC"
                        ))

        return edits

# --------------------------
# 2) SpellChecker + medical guard
# --------------------------
class SpellChecker(BaseExtractorModule):
    def __init__(self, medical_terms_manager=None):
        self.medical_terms_manager = medical_terms_manager
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        # Optional: load large dictionary, e.g. sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

    def is_medical_term(self, token: str) -> bool:
        if self.medical_terms_manager:
            return self.medical_terms_manager.is_medical_term(token)
        return False

    def detect(self, text: str) -> List[SpanEdit]:
        edits = []
        for m in re.finditer(r"\b[A-Za-z0-9'-]+\b", text):
            token = m.group(0)
            if self.is_medical_term(token):
                continue
            suggestions = self.sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions and suggestions[0].term != token:
                edits.append(SpanEdit(
                    start_char=m.start(),
                    end_char=m.end(),
                    op="REPLACE",
                    cand_texts=[s.term for s in suggestions[:3]],
                    score=1 - suggestions[0].distance/3,
                    tag="SPL",
                    edit_type="deterministic",
                    detector_name="SpellChecker"
                ))
        return edits

# --------------------------
# 3) Repetition detector
# --------------------------
class RepetitionDetector(BaseExtractorModule):
    def __init__(self, max_ngram=3):
        self.max_ngram = max_ngram

    def detect(self, text: str) -> List[SpanEdit]:
        edits = []
        tokens = re.findall(r"\S+", text)
        if not tokens:
            return edits

        char_positions = []
        idx = 0
        for t in tokens:
            start = text.find(t, idx)
            end = start + len(t)
            char_positions.append((t, start, end))
            idx = end

        for n in range(1, self.max_ngram + 1):
            for i in range(len(tokens) - 2*n + 1):
                a = tokens[i:i+n]
                b = tokens[i+n:i+2*n]
                if a == b:
                    start = char_positions[i+n][1]
                    end = char_positions[i+2*n-1][2]
                    edits.append(SpanEdit(
                        start, end, "DELETE", [""], 0.9,
                        tag="RPT", edit_type="deterministic", detector_name="Repetition"
                    ))
        return edits

# --------------------------
# 4) WSD / Ambiguity detector (GlossBERT-based)
# --------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from typing import List, Tuple

class CombinedMedicalDetector(BaseExtractorModule):
    """
    Combined LLM-based detector for both ambiguity detection and non-medical dialogue detection.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gpt-4o-mini",
        base_url: str = None,
        api_timeout: float = None,
        api_config: dict = None,
    ):
        super().__init__()
        self.api_config = api_config
        if api_key:
            self.client = create_openai_client(
                api_key,
                base_url,
                timeout=api_timeout,
                config=api_config,
            )
            self.model_name = model_name
        else:
            self.client = None
            self.model_name = model_name

    def _detect_with_llm(self, text: str) -> str:
        """Detect ambiguous terms and non-medical dialogue with LLM."""
        if not self.client:
            return text
            
        system_prompt = PCRAgentPrompts.combined_detector_system()
        user_prompt = PCRAgentPrompts.combined_detector_user(text)

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = chat_completions_create(
                self.client,
                context="PCRAgent CombinedMedicalDetector ambiguity/NOS detection",
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=None,
                api_config=self.api_config,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print_api_error(
                e,
                context="PCRAgent CombinedMedicalDetector (fallback: return original text)",
                model=self.model_name,
                show_traceback=True,
            )
            return text

    def _extract_ambiguity_tags(self, annotated_text: str) -> List[Tuple[str, int, int]]:
        """Extract ambiguity tag information from annotated text."""
        ambiguity_tags = []
        pattern = PCRAgentPrompts.AMBIGUITY_TAG_PATTERN

        for match in re.finditer(pattern, annotated_text):
            original_word = match.group(1)
            start_pos = match.start()
            end_pos = match.end()
            ambiguity_tags.append((original_word, start_pos, end_pos))
            
        return ambiguity_tags

    def _extract_non_medical_fragments(self, annotated_text: str) -> List[Tuple[str, int, int]]:
        """Extract non-medical dialogue fragments from annotated text."""
        non_medical_fragments = []
        pattern = r'\[NOS:start\](.*?)\[NOS:end\]'
        
        for match in re.finditer(pattern, annotated_text, re.DOTALL):
            fragment = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            non_medical_fragments.append((fragment, start_pos, end_pos))
            
        return non_medical_fragments

    def _protect_existing_tags(self, text: str) -> str:
        """Protect existing tags to avoid duplicate processing."""
        # Remove existing AMBIG and NOS tags to avoid duplicate annotation
        protected_text = re.sub(PCRAgentPrompts.AMBIGUITY_TAG_PATTERN, '', text)
        protected_text = re.sub(r'\[NOS:start\].*?\[NOS:end\]', '', protected_text, flags=re.DOTALL)
        return protected_text

    def detect(self, text: str) -> List[SpanEdit]:
        """Detect ambiguous terms and non-medical dialogue with LLM."""
        if not self.client:
            return []
            
        # Preprocessing: strip existing tags
        clean_text = self._protect_existing_tags(text)

        # Use LLM to detect ambiguous terms and non-medical dialogue
        annotated_text = self._detect_with_llm(clean_text)

        edits = []

        # Process ambiguity tags
        ambiguity_tags = self._extract_ambiguity_tags(annotated_text)
        for original_word, tag_start, tag_end in ambiguity_tags:
            # Locate the word in the original text
            word_pattern = re.escape(original_word)
            for match in re.finditer(word_pattern, text, re.IGNORECASE):
                word_start = match.start()
                word_end = match.end()
                
                # Skip if already covered by another edit
                is_covered = False
                for existing_edit in edits:
                    if (word_start >= existing_edit.start_char and word_start < existing_edit.end_char) or \
                       (word_end > existing_edit.start_char and word_end <= existing_edit.end_char):
                        is_covered = True
                        break
                
                if not is_covered:
                    edits.append(SpanEdit(
                        start_char=word_start,
                        end_char=word_end,
                        op="REPLACE",
                        cand_texts=[f"[AMBIG:{original_word}]"],
                        score=0.8,  # Default confidence
                        tag="AMB",
                        edit_type="candidate",
                        detector_name="CombinedMedical"
                    ))
                    break  # Process only the first match

        # Process non-medical dialogue fragments
        non_medical_fragments = self._extract_non_medical_fragments(annotated_text)
        for fragment, tag_start, tag_end in non_medical_fragments:
            if fragment and fragment in text:
                start = text.find(fragment)
                if start != -1:
                    edits.append(SpanEdit(
                        start_char=start,
                        end_char=start + len(fragment),
                        op="DELETE",
                        cand_texts=[""],
                        score=0.9,
                        tag="NOS",
                        edit_type="deterministic",
                        detector_name="CombinedMedical"
                    ))
        
        return edits


# -------------------------- Editor Pipeline --------------------------
class EditManager:
    """Editor stage: Classify, score, and filter candidate edits"""
    
    def __init__(self, w1=0.3, w2=0.4, w3=0.3, delta=0.05):
        self.w1 = w1  # Edit cost weight
        self.w2 = w2  # Fluency weight  
        self.w3 = w3  # WSD confidence weight
        self.delta = delta  # Threshold for multi-candidate retention
        
        # Edit priority (lower number = higher priority)
        # Fixed order: spell, repetition, grammar, ambiguity, nonmedical
        self.edit_priority = {
            "SPL": 1,  # Spelling errors - highest priority (step 1)
            "RPT": 2,  # Repetition - high priority (step 2)
            "GRM": 3,  # Grammar errors - medium-high priority (step 3)
            "AMB": 4,  # Ambiguity resolution - medium priority (step 4)
            "NOS": 5   # Non-medical content - low priority (step 5)
        }

        # Edit-type compatibility matrix
        self.compatibility_matrix = {
            ("GRM", "SPL"): True,   # Grammar and spelling can merge
            ("GRM", "AMB"): False,  # Grammar and ambiguity incompatible
            ("SPL", "AMB"): False,  # Spelling and ambiguity incompatible
            ("RPT", "GRM"): True,   # Repetition and grammar can merge
            ("RPT", "SPL"): True,   # Repetition and spelling can merge
            ("NOS", "GRM"): False,  # Non-medical dialogue incompatible with grammar
            ("NOS", "SPL"): False,  # Non-medical dialogue incompatible with spelling
        }
        
    def classify_edits(self, edits: List[SpanEdit]) -> Tuple[List[SpanEdit], List[SpanEdit]]:
        """Classify edits into deterministic vs candidate"""

        deterministic = []
        candidates = []
        
        for edit in edits:
            if edit.tag in ["RPT", "SPL", "GRM", "NOS", "AMB"]:
                edit.edit_type = "deterministic"
                deterministic.append(edit)
            else:
                # Default to deterministic for unknown tags
                edit.edit_type = "deterministic"
                deterministic.append(edit)
                
        return deterministic, candidates
    
    def calculate_edit_cost(self, original: str, candidate: str) -> float:
        """Calculate normalized edit cost (0-1, lower is better)"""
        if not original and not candidate:
            return 0.0
        if not original:
            return 1.0
        if not candidate:
            return 1.0
            
        # Levenshtein distance normalized by max length
        max_len = max(len(original), len(candidate))
        if max_len == 0:
            return 0.0
            
        # Simple character-level edit distance
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein(original, candidate)
        return distance / max_len
    
    def calculate_fluency_score(self, text: str) -> float:
        """Calculate pseudo-perplexity based fluency score"""
        # Simple heuristic: longer sentences with proper punctuation are more fluent
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0.0
            
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        punctuation_score = len(re.findall(r'[.!?]', text)) / max(1, len(sentences))
        
        # Normalize to 0-1 range
        fluency = min(1.0, (avg_length / 10.0) * 0.5 + punctuation_score * 0.5)
        return fluency
    
    def calculate_wsd_confidence(self, edit: SpanEdit) -> float:
        """Calculate WSD confidence for ambiguous edits"""
        if edit.tag != "AMB":
            return 1.0
            
        # For AMB edits, use the score as confidence
        return edit.score
    
    def score_candidates(self, edit: SpanEdit, context: str) -> List[Tuple[str, float]]:
        """Score all candidates for an edit"""
        if not edit.cand_texts:
            return []
            
        original_text = context[edit.start_char:edit.end_char]
        scored_candidates = []
        
        for candidate in edit.cand_texts:
            # Calculate individual scores
            edit_cost = self.calculate_edit_cost(original_text, candidate)
            fluency = self.calculate_fluency_score(candidate)
            wsd_conf = self.calculate_wsd_confidence(edit)
            
            # Combined score
            score = (self.w1 * (1 - edit_cost) + 
                    self.w2 * fluency + 
                    self.w3 * wsd_conf)
            
            scored_candidates.append((candidate, score))
        
        return scored_candidates
    
    def filter_candidates(self, edit: SpanEdit, context: str) -> SpanEdit:
        """Filter and rank candidates based on scoring"""
        if edit.edit_type == "deterministic":
            return edit
            
        scored_candidates = self.score_candidates(edit, context)
        if not scored_candidates:
            return edit
            
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Check if top candidates are close in score
        if len(scored_candidates) > 1:
            top_score = scored_candidates[0][1]
            second_score = scored_candidates[1][1]
            
            if top_score - second_score < self.delta:
                # Keep multiple candidates for arbiter
                edit.cand_texts = [cand for cand, _ in scored_candidates[:3]]
            else:
                # Keep only top candidate
                edit.cand_texts = [scored_candidates[0][0]]
        else:
            edit.cand_texts = [scored_candidates[0][0]]
            
        return edit
    
    def merge_overlapping_edits(self, edits: List[SpanEdit]) -> List[SpanEdit]:
        if not edits:
            return []
            
        # Sort by priority and position
        sorted_edits = sorted(edits, key=lambda x: (self.edit_priority.get(x.tag, 999), x.start_char))
        merged = []
        
        for edit in sorted_edits:
            if not merged:
                merged.append(edit)
                continue
                
            last_edit = merged[-1]
            
            # Check overlap
            if self._positions_overlap(edit, last_edit):
                # Smart merge strategy
                merged_edit = self._smart_merge_edits(edit, last_edit)
                merged[-1] = merged_edit
            else:
                merged.append(edit)
                
        return merged
    
    def _smart_merge_edits(self, edit1: SpanEdit, edit2: SpanEdit) -> SpanEdit:
        """Intelligently merge two edits."""
        # Check compatibility
        if not self._check_compatibility(edit1, edit2):
            # If incompatible, keep higher-priority edit
            winner = self._get_priority_winner(edit1, edit2)
            return winner
        
        # Merge when compatible
        merged_candidates = self._merge_candidates(edit1, edit2)
        merged_score = self._calculate_merged_score(edit1, edit2)
        merged_tags = self._merge_tags(edit1.tag, edit2.tag)
        
        return SpanEdit(
            start_char=min(edit1.start_char, edit2.start_char),
            end_char=max(edit1.end_char, edit2.end_char),
            op=self._determine_merged_operation(edit1, edit2),
            cand_texts=merged_candidates,
            score=merged_score,
            tag=merged_tags,
            edit_type="candidate",
            detector_name=f"{edit1.detector_name}+{edit2.detector_name}"
        )
    
    def _merge_candidates(self, edit1: SpanEdit, edit2: SpanEdit) -> List[str]:
        """Intelligently merge candidate texts."""
        candidates1 = set(edit1.cand_texts)
        candidates2 = set(edit2.cand_texts)

        # Return directly if candidates are identical
        if candidates1 == candidates2:
            return list(candidates1)
        
        # Merge candidates, deduplicate, preserve order
        merged = []
        for cand in edit1.cand_texts:
            if cand not in merged:
                merged.append(cand)
        for cand in edit2.cand_texts:
            if cand not in merged:
                merged.append(cand)
        
        # Cap candidate count
        return merged[:5]
    
    def _calculate_merged_score(self, edit1: SpanEdit, edit2: SpanEdit) -> float:
        """Calculate merged score."""
        # Weighted average based on priority
        priority1 = self.edit_priority.get(edit1.tag, 999)
        priority2 = self.edit_priority.get(edit2.tag, 999)

        # Higher priority => larger weight
        weight1 = 1.0 / priority1
        weight2 = 1.0 / priority2
        total_weight = weight1 + weight2
        
        return (weight1 * edit1.score + weight2 * edit2.score) / total_weight
    
    def _merge_tags(self, tag1: str, tag2: str) -> str:
        """Intelligently merge tags."""
        if tag1 == tag2:
            return tag1

        # Sort tags by priority
        tags = [tag1, tag2]
        tags.sort(key=lambda x: self.edit_priority.get(x, 999))
        return "+".join(tags)
    
    def _determine_merged_operation(self, edit1: SpanEdit, edit2: SpanEdit) -> str:
        """Determine merged operation type."""
        # Keep DELETE if both are DELETE
        if edit1.op == "DELETE" and edit2.op == "DELETE":
            return "DELETE"
        # Keep INSERT if both are INSERT
        elif edit1.op == "INSERT" and edit2.op == "INSERT":
            return "INSERT"
        # Otherwise use REPLACE
        else:
            return "REPLACE"
    
    def process_edits(self, edits: List[SpanEdit], text: str) -> List[SpanEdit]:
        """Main processing pipeline for Editor stage"""
        # Classify edits (all are deterministic now)
        deterministic, candidates = self.classify_edits(edits)
        
        # Simplified: skip scoring/filtering; merge all edits directly
        all_processed = deterministic + candidates
        merged_edits = self.merge_overlapping_edits(all_processed)
        
        return merged_edits
    
    def _positions_overlap(self, edit1: SpanEdit, edit2: SpanEdit) -> bool:
        """Check if two edits have overlapping positions"""
        return (edit1.start_char < edit2.end_char and 
                edit2.start_char < edit1.end_char)
    
    def _check_compatibility(self, edit1: SpanEdit, edit2: SpanEdit) -> bool:
        """Check whether two edits are compatible."""
        tag1, tag2 = edit1.tag, edit2.tag

        # Consult compatibility matrix
        if (tag1, tag2) in self.compatibility_matrix:
            return self.compatibility_matrix[(tag1, tag2)]
        if (tag2, tag1) in self.compatibility_matrix:
            return self.compatibility_matrix[(tag2, tag1)]
        
        # Default compatibility rules
        if tag1 == tag2:
            return True  # Same tag type is compatible

        # Different priority types are usually incompatible
        return False
    
    def _get_priority_winner(self, edit1: SpanEdit, edit2: SpanEdit) -> SpanEdit:
        """Select winning edit by priority rules."""
        priority1 = self.edit_priority.get(edit1.tag, 999)
        priority2 = self.edit_priority.get(edit2.tag, 999)

        if priority1 < priority2:
            return edit1
        elif priority2 < priority1:
            return edit2
        else:
            # Tie-break by higher score
            return edit1 if edit1.score >= edit2.score else edit2
    
    def _merge_tags(self, tag1: str, tag2: str) -> str:
        """Merge tags."""
        if tag1 == tag2:
            return tag1

        # Choose primary tag by priority
        priority1 = self.edit_priority.get(tag1, 999)
        priority2 = self.edit_priority.get(tag2, 999)
        
        if priority1 <= priority2:
            return tag1
        else:
            return tag2
    
    def _determine_merged_operation(self, edit1: SpanEdit, edit2: SpanEdit) -> str:
        """Determine merged operation type."""
        # Both DELETE => merged DELETE
        if edit1.op == "DELETE" and edit2.op == "DELETE":
            return "DELETE"

        # Both INSERT => merged INSERT
        if edit1.op == "INSERT" and edit2.op == "INSERT":
            return "INSERT"

        # Default to REPLACE otherwise
        return "REPLACE"

class EditorPipeline:
    """Editor stage pipeline"""
    
    def __init__(
        self,
        w1=0.3,
        w2=0.4,
        w3=0.3,
        delta=0.05,
        api_key: str = None,
        base_url: str = None,
        api_timeout: float = None,
        api_config: dict = None,
    ):
        self.edit_manager = EditManager(w1, w2, w3, delta)
        self.api_key = api_key
        self.base_url = base_url
        self.api_timeout = api_timeout
        self.api_config = api_config
        
    def _interpret_ambiguity_word(self, word: str, context: str) -> str:
        """Generate English medical gloss for ambiguous term; return word(interpretation)."""
        if not self.api_key:
            return word
            
        client = create_openai_client(
            self.api_key,
            self.base_url,
            timeout=self.api_timeout,
            config=self.api_config,
        )

        system_prompt = PCRAgentPrompts.ambiguity_interpretation_system()
        user_prompt = PCRAgentPrompts.ambiguity_interpretation_user(word, context)

        try:
            response = chat_completions_create(
                client,
                context="PCRAgent EditorPipeline ambiguity interpretation",
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=50,
            )
            interpretation = response.choices[0].message.content.strip()
            if interpretation.startswith('"') and interpretation.endswith('"'):
                interpretation = interpretation[1:-1]
            if interpretation.startswith("'") and interpretation.endswith("'"):
                interpretation = interpretation[1:-1]
            if "interpretation:" in interpretation.lower():
                interpretation = interpretation.split(":")[-1].strip()
            return f"{word}({interpretation})" if interpretation else word
        except Exception as e:
            print_api_error(
                e,
                context="PCRAgent EditorPipeline ambiguity interpretation (fallback: keep original word)",
                model="gpt-4o-mini",
                show_traceback=True,
            )
            return word
    
    def run(self, edits: List[SpanEdit], text: str) -> Dict:
        """Run editor pipeline"""
        # Process edits: merge overlapping edits
        processed_edits = self.edit_manager.process_edits(edits, text)

        # Process AMB edits: convert [AMBIG:word] to word(interpretation)
        for edit in processed_edits:
            if edit.tag == "AMB" and edit.cand_texts:
                # Extract original word from [AMBIG:word] tag
                import re
                amb_tag = edit.cand_texts[0] if edit.cand_texts else ""
                match = re.match(r'\[(?:AMB|AMBIG|AMG):([^\]]+)\]', amb_tag)
                if match:
                    original_word = match.group(1)
                    # Extract sentence context containing the word
                    sentence_start = max(0, text.rfind('.', 0, edit.start_char) + 1, 
                                        text.rfind('!', 0, edit.start_char) + 1,
                                        text.rfind('?', 0, edit.start_char) + 1)
                    sentence_end = len(text)
                    for punct in ['.', '!', '?']:
                        end_pos = text.find(punct, edit.end_char)
                        if end_pos != -1:
                            sentence_end = min(sentence_end, end_pos + 1)
                    context = text[sentence_start:sentence_end].strip()
                    
                    # Generate annotated format
                    annotated_word = self._interpret_ambiguity_word(original_word, context)
                    edit.cand_texts = [annotated_word]
        
        # All edits are deterministic; apply directly
        edited_text = self.apply_deterministic_edits(text, processed_edits)

        return {
            "processed_edits": processed_edits,
            "deterministic_edits": processed_edits,  # All edits are deterministic
            "candidate_edits": [],  # No candidate edits
            "edited_text": edited_text,
            "interpreted_text": text  # interpreted_text no longer handled separately
        }
    
    def apply_deterministic_edits(self, text: str, edits: List[SpanEdit]) -> str:
        """Apply deterministic edits to text"""
        if not edits:
            return text
            
        # Sort edits by position (reverse order to maintain indices)
        sorted_edits = sorted(edits, key=lambda x: x.start_char, reverse=True)
        
        result = text
        for edit in sorted_edits:
            start = edit.start_char
            end = edit.end_char
            
            if edit.op == "DELETE":
                result = result[:start] + result[end:]
            elif edit.op == "REPLACE":
                replacement = edit.cand_texts[0] if edit.cand_texts else ""
                result = result[:start] + replacement + result[end:]
            elif edit.op == "INSERT":
                insertion = edit.cand_texts[0] if edit.cand_texts else ""
                result = result[:start] + insertion + result[start:]
                
        return result

# -------------------------- Arbiter Pipeline --------------------------
class ArbiterCore:
    """Arbiter core: Conflict detection and candidate evaluation"""
    
    def __init__(self):
        self.conflict_threshold = 0.1  # Threshold for score differences
        
        # Edit priority rules (lower number = higher priority)
        # Fixed order: spell, repetition, grammar, ambiguity, nonmedical
        self.edit_priority = {
            "SPL": 1,  # Spelling errors - highest priority (step 1)
            "RPT": 2,  # Repetition - high priority (step 2)
            "GRM": 3,  # Grammar errors - medium-high priority (step 3)
            "AMB": 4,  # Ambiguity resolution - medium priority (step 4)
            "NOS": 5,  # Non-medical dialogue - low priority (step 5)
            "Fragment": 6  # Fragment noise - lowest priority (removed; kept for compatibility)
        }

        # Edit-type compatibility matrix
        self.compatibility_matrix = {
            ("GRM", "SPL"): True,   # Grammar and spelling can merge
            ("GRM", "AMB"): False,  # Grammar and ambiguity incompatible
            ("SPL", "AMB"): False,  # Spelling and ambiguity incompatible
            ("RPT", "GRM"): True,   # Repetition and grammar can merge
            ("RPT", "SPL"): True,   # Repetition and spelling can merge
            ("NOS", "GRM"): False,  # Non-medical dialogue incompatible with grammar
            ("NOS", "SPL"): False,  # Non-medical dialogue incompatible with spelling
        }
        
    def detect_conflicts(self, edits: List[SpanEdit]) -> List[Dict]:
        """Detect position and candidate conflicts with enhanced analysis"""
        conflicts = []
        
        # Check for position conflicts with priority analysis
        for i, edit1 in enumerate(edits):
            for j, edit2 in enumerate(edits[i+1:], i+1):
                if self._positions_overlap(edit1, edit2):
                    # Analyze conflict type and priority
                    conflict_type = self._analyze_position_conflict(edit1, edit2)
                    conflicts.append({
                        "type": "position_conflict",
                        "edit1": edit1,
                        "edit2": edit2,
                        "overlap_span": (max(edit1.start_char, edit2.start_char),
                                       min(edit1.end_char, edit2.end_char)),
                        "conflict_type": conflict_type,
                        "priority_winner": self._get_priority_winner(edit1, edit2),
                        "compatibility": self._check_compatibility(edit1, edit2)
                    })
        
        # Check for candidate conflicts
        for edit in edits:
            if len(edit.cand_texts) > 1:
                conflicts.append({
                    "type": "candidate_conflict", 
                    "edit": edit,
                    "candidates": edit.cand_texts,
                    "candidate_scores": self._score_all_candidates(edit)
                })
                
        return conflicts
    
    def _positions_overlap(self, edit1: SpanEdit, edit2: SpanEdit) -> bool:
        """Check if two edits have overlapping positions"""
        return (edit1.start_char < edit2.end_char and 
                edit2.start_char < edit1.end_char)
    
    def _analyze_position_conflict(self, edit1: SpanEdit, edit2: SpanEdit) -> str:
        """Analyze position conflict type."""
        # Check exact overlap
        if (edit1.start_char == edit2.start_char and edit1.end_char == edit2.end_char):
            return "exact_overlap"
        # Check containment
        elif (edit1.start_char <= edit2.start_char and edit1.end_char >= edit2.end_char):
            return "edit1_contains_edit2"
        elif (edit2.start_char <= edit1.start_char and edit2.end_char >= edit1.end_char):
            return "edit2_contains_edit1"
        # Check partial overlap
        else:
            return "partial_overlap"
    
    def _get_priority_winner(self, edit1: SpanEdit, edit2: SpanEdit) -> SpanEdit:
        """Select winning edit by priority rules."""
        priority1 = self.edit_priority.get(edit1.tag, 999)
        priority2 = self.edit_priority.get(edit2.tag, 999)

        if priority1 < priority2:
            return edit1
        elif priority2 < priority1:
            return edit2
        else:
            # Tie-break by higher score
            return edit1 if edit1.score >= edit2.score else edit2
    
    def _check_compatibility(self, edit1: SpanEdit, edit2: SpanEdit) -> bool:
        """Check whether two edits are compatible."""
        tag1, tag2 = edit1.tag, edit2.tag

        # Check direct compatibility
        if (tag1, tag2) in self.compatibility_matrix:
            return self.compatibility_matrix[(tag1, tag2)]
        elif (tag2, tag1) in self.compatibility_matrix:
            return self.compatibility_matrix[(tag2, tag1)]
        
        # Default compatibility rules
        return tag1 == tag2  # Same tag is compatible by default

    def _score_all_candidates(self, edit: SpanEdit) -> Dict[str, float]:
        """Score all candidates for an edit."""
        scores = {}
        for candidate in edit.cand_texts:
            # Use simplified scoring
            score = self._calculate_comprehensive_score(edit, candidate, "")
            scores[candidate] = score
        return scores
    
    def evaluate_candidates(self, edit: SpanEdit, context: str) -> Dict:
        """Evaluate all candidates for an edit"""
        if not edit.cand_texts:
            return {"best_candidate": "", "scores": {}}
            
        scores = {}
        for candidate in edit.cand_texts:
            score = self._calculate_comprehensive_score(edit, candidate, context)
            scores[candidate] = score
            
        # Find best candidate
        best_candidate = max(scores.keys(), key=lambda x: scores[x])
        
        return {
            "best_candidate": best_candidate,
            "scores": scores,
            "confidence": scores[best_candidate]
        }
    
    def _calculate_comprehensive_score(self, edit: SpanEdit, candidate: str, context: str) -> float:
        """Calculate comprehensive score for a candidate"""
        # Edit cost (lower is better)
        original_text = context[edit.start_char:edit.end_char]
        edit_cost = self._calculate_edit_cost(original_text, candidate)
        
        # Fluency score
        fluency = self._calculate_fluency(candidate)
        
        # Semantic consistency (simplified)
        semantic_consistency = self._calculate_semantic_consistency(original_text, candidate, context)
        
        # Term preservation (for medical terms)
        term_preservation = self._calculate_term_preservation(original_text, candidate)
        
        # Repetition penalty
        repetition_penalty = self._calculate_repetition_penalty(candidate, context)
        
        # Weighted combination
        score = (0.2 * (1 - edit_cost) + 
                0.3 * fluency + 
                0.2 * semantic_consistency + 
                0.2 * term_preservation + 
                0.1 * (1 - repetition_penalty))
        
        return max(0.0, min(1.0, score))
    
    def _calculate_edit_cost(self, original: str, candidate: str) -> float:
        """Calculate normalized edit cost"""
        if not original and not candidate:
            return 0.0
        max_len = max(len(original), len(candidate))
        if max_len == 0:
            return 0.0
            
        # Simple character-level distance
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        distance = levenshtein(original, candidate)
        return distance / max_len
    
    def _calculate_fluency(self, text: str) -> float:
        """Calculate fluency score"""
        if not text:
            return 0.0
            
        # Simple heuristics
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]', text)) + 1
        
        # Average sentence length
        avg_sentence_length = word_count / sentence_count
        
        # Punctuation usage
        punctuation_score = len(re.findall(r'[.!?,;:]', text)) / max(1, word_count)
        
        # Combine factors
        fluency = min(1.0, (avg_sentence_length / 15.0) * 0.7 + punctuation_score * 10 * 0.3)
        return fluency
    
    def _calculate_semantic_consistency(self, original: str, candidate: str, context: str) -> float:
        """Calculate semantic consistency score"""
        # Simplified: check if candidate maintains similar word patterns
        original_words = set(original.lower().split())
        candidate_words = set(candidate.lower().split())
        
        if not original_words and not candidate_words:
            return 1.0
        if not original_words or not candidate_words:
            return 0.5
            
        # Jaccard similarity
        intersection = len(original_words & candidate_words)
        union = len(original_words | candidate_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_term_preservation(self, original: str, candidate: str) -> float:
        """Calculate medical term preservation score"""
        # Simple check for medical-like terms (capitalized words, common medical suffixes)
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:itis|osis|emia|uria|algia|pathy)\b',  # Medical suffixes
            r'\b(?:heart|brain|liver|lung|kidney|blood|pain|fever|temperature)\b',  # Common medical terms
        ]
        
        original_medical = sum(len(re.findall(pattern, original, re.IGNORECASE)) for pattern in medical_patterns)
        candidate_medical = sum(len(re.findall(pattern, candidate, re.IGNORECASE)) for pattern in medical_patterns)
        
        if original_medical == 0:
            return 1.0 if candidate_medical == 0 else 0.8
            
        preservation_ratio = candidate_medical / original_medical
        return min(1.0, preservation_ratio)
    
    def _calculate_repetition_penalty(self, candidate: str, context: str) -> float:
        """Calculate repetition penalty"""
        if not candidate:
            return 0.0
            
        # Check for repeated words in candidate
        words = candidate.split()
        if len(words) <= 1:
            return 0.0
            
        # Count repeated words
        word_counts = Counter(words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        
        # Normalize by total words
        repetition_ratio = repeated_words / len(words)
        return min(1.0, repetition_ratio)

class ArbiterPipeline:
    """Enhanced Arbiter stage pipeline with intelligent conflict resolution"""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        api_timeout: float = None,
        api_config: dict = None,
    ):
        self.arbiter_core = ArbiterCore()
        self.api_key = api_key
        self.base_url = base_url
        self.api_timeout = api_timeout
        self.api_config = api_config
    
    def run(self, edits: List[SpanEdit], original_text: str, editor_processed_text: str) -> Dict:
        """
        Run enhanced arbiter pipeline with intelligent conflict resolution
        
        Args:
            edits: List of edits
            original_text: Original input sentence
            editor_processed_text: Sentence after Editor processing
        """
        # Detect conflicts with enhanced analysis
        conflicts = self.arbiter_core.detect_conflicts(edits)
        
        # Resolve conflicts using intelligent strategies
        resolved_edits = self._resolve_conflicts_intelligently(edits, editor_processed_text, conflicts)
        
        # Apply all resolved edits
        edited_text = self.apply_resolved_edits(editor_processed_text, resolved_edits)
        
        # Use LLM to check and correct the editor-processed sentence
        final_text = self._check_and_correct_editor_output(original_text, editor_processed_text, edited_text)
        
        return {
            "conflicts": conflicts,
            "resolved_edits": resolved_edits,
            "final_text": final_text,
            "edited_text": edited_text,
            "resolution_strategy": "intelligent_priority_based"
        }
    
    def _resolve_conflicts_intelligently(self, edits: List[SpanEdit], text: str, conflicts: List[Dict]) -> List[SpanEdit]:
        """Intelligent conflict resolution strategy."""
        # Build edit map to track processing state
        edit_map = {id(edit): edit for edit in edits}
        processed_edits = set()
        resolved_edits = []

        # Sort edits by priority
        sorted_edits = sorted(edits, key=lambda x: (
            self.arbiter_core.edit_priority.get(x.tag, 999),
            -x.score,  # Higher score first
            x.start_char  # Earlier position first
        ))
        
        for edit in sorted_edits:
            if id(edit) in processed_edits:
                continue
                
            # Check for position conflicts
            conflicting_edits = self._find_conflicting_edits(edit, edit_map, processed_edits)

            if conflicting_edits:
                # Resolve conflict
                resolved_edit = self._resolve_position_conflicts(edit, conflicting_edits, text)
                resolved_edits.append(resolved_edit)

                # Mark conflicting edits as processed
                processed_edits.add(id(edit))
                for conflict_edit in conflicting_edits:
                    processed_edits.add(id(conflict_edit))
            else:
                # No conflict; process directly
                resolved_edit = self._resolve_single_edit(edit, text)
                resolved_edits.append(resolved_edit)
                processed_edits.add(id(edit))
        
        return resolved_edits
    
    def _find_conflicting_edits(self, edit: SpanEdit, edit_map: Dict, processed_edits: set) -> List[SpanEdit]:
        """Find edits conflicting with the given edit."""
        conflicting = []
        for other_edit in edit_map.values():
            if (id(other_edit) != id(edit) and 
                id(other_edit) not in processed_edits and
                self.arbiter_core._positions_overlap(edit, other_edit)):
                conflicting.append(other_edit)
        return conflicting
    
    def _resolve_position_conflicts(self, primary_edit: SpanEdit, conflicting_edits: List[SpanEdit], text: str) -> SpanEdit:
        """Resolve position conflicts."""
        # Check compatibility
        compatible_edits = [primary_edit]
        for conflict_edit in conflicting_edits:
            if self.arbiter_core._check_compatibility(primary_edit, conflict_edit):
                compatible_edits.append(conflict_edit)

        if len(compatible_edits) > 1:
            # Compatible edits can be merged
            return self._merge_compatible_edits(compatible_edits, text)
        else:
            # Incompatible: keep highest-priority edit
            all_edits = [primary_edit] + conflicting_edits
            winner = self.arbiter_core._get_priority_winner(primary_edit, conflicting_edits[0])
            return self._resolve_single_edit(winner, text)
    
    def _merge_compatible_edits(self, compatible_edits: List[SpanEdit], text: str) -> SpanEdit:
        """Merge compatible edits."""
        if len(compatible_edits) == 1:
            return self._resolve_single_edit(compatible_edits[0], text)

        # Merge all compatible edits
        merged_candidates = []
        merged_score = 0
        merged_tags = []

        for edit in compatible_edits:
            merged_candidates.extend(edit.cand_texts)
            merged_score += edit.score
            merged_tags.append(edit.tag)

        # Deduplicate candidate texts
        merged_candidates = list(set(merged_candidates))

        # Compute average score
        merged_score = merged_score / len(compatible_edits)

        # Merge tags
        merged_tags = "+".join(sorted(set(merged_tags)))

        # Create merged edit
        merged_edit = SpanEdit(
            start_char=min(e.start_char for e in compatible_edits),
            end_char=max(e.end_char for e in compatible_edits),
            op="REPLACE",
            cand_texts=merged_candidates,
            score=merged_score,
            tag=merged_tags,
            edit_type="candidate",
            detector_name="+".join(set(e.detector_name for e in compatible_edits))
        )
        
        return self._resolve_single_edit(merged_edit, text)
    
    def _resolve_single_edit(self, edit: SpanEdit, text: str) -> SpanEdit:
        """Resolve a single edit."""
        if edit.edit_type == "deterministic":
            return edit

        # Evaluate candidates
        evaluation = self.arbiter_core.evaluate_candidates(edit, text)

        # Select best candidate from evaluation
        best_candidate = evaluation["best_candidate"]

        # Create final edit
        return SpanEdit(
            start_char=edit.start_char,
            end_char=edit.end_char,
            op=edit.op,
            cand_texts=[best_candidate],
            score=evaluation["confidence"],
            tag=edit.tag,
            edit_type="deterministic",
            detector_name=edit.detector_name
        )
    
    def apply_resolved_edits(self, text: str, edits: List[SpanEdit]) -> str:
        """Apply all resolved edits to produce final text"""
        if not edits:
            return text
            
        # Sort edits by position (reverse order to maintain indices)
        sorted_edits = sorted(edits, key=lambda x: x.start_char, reverse=True)
        
        result = text
        for edit in sorted_edits:
            start = edit.start_char
            end = edit.end_char
            
            if edit.op == "DELETE":
                result = result[:start] + result[end:]
            elif edit.op == "REPLACE":
                replacement = edit.cand_texts[0] if edit.cand_texts else ""
                result = result[:start] + replacement + result[end:]
            elif edit.op == "INSERT":
                insertion = edit.cand_texts[0] if edit.cand_texts else ""
                result = result[:start] + insertion + result[start:]
                
        return result
    
    def _check_and_correct_editor_output(self, original_text: str, editor_processed_text: str, edited_text: str) -> str:
        """
        Verify Editor output against the original sentence and correct if needed.

        Args:
            original_text: Original input sentence
            editor_processed_text: Sentence after Editor processing
            edited_text: Text after applying arbiter-resolved edits

        Returns:
            str: Corrected final text
        """
        if not self.api_key:
            return edited_text
        
        client = create_openai_client(
            self.api_key,
            self.base_url,
            timeout=self.api_timeout,
            config=self.api_config,
        )

        system_prompt = PCRAgentPrompts.arbiter_check_system()
        user_prompt = PCRAgentPrompts.arbiter_check_user(
            original_text, editor_processed_text, edited_text
        )
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            out_tokens = estimate_max_tokens(
                original_text,
                editor_processed_text,
                edited_text,
                system_prompt,
                user_prompt,
                config=self.api_config,
            )
            response = chat_completions_create(
                client,
                context="PCRAgent ArbiterPipeline denoising QA",
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=out_tokens,
                api_config=self.api_config,
            )
            result = response.choices[0].message.content.strip()
            if "Corrected sentence:" in result:
                result = result.split("Corrected sentence:")[-1].strip()
            if "Final sentence:" in result:
                result = result.split("Final sentence:")[-1].strip()
            if "The corrected sentence is:" in result:
                result = result.split("The corrected sentence is:")[-1].strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            if result.startswith("'") and result.endswith("'"):
                result = result[1:-1]
            result = re.sub(PCRAgentPrompts.AMBIGUITY_TAG_PATTERN, "", result)
            result = re.sub(r"\[NOS:start\].*?\[NOS:end\]", "", result, flags=re.DOTALL)
            return result if result else edited_text
        except Exception as e:
            print_api_error(
                e,
                context="PCRAgent ArbiterPipeline denoising QA (fallback: return current text)",
                model="gpt-4o-mini",
                show_traceback=True,
            )
            return edited_text

# -------------------------- Evaluation Metrics --------------------------
class MedicalTermsManager:
    """Medical dictionary manager for efficient term lookup."""
    
    def __init__(self, dictionary_path: str = None):
        self.dictionary_path = dictionary_path
        self._terms_set = None
        self._terms_trie = None
        self._regex_patterns_cache = {}
        self._loaded = False
        
    def load_medical_dictionary(self, dictionary_path: str = None) -> bool:
        """Load medical dictionary file."""
        if dictionary_path:
            self.dictionary_path = dictionary_path

        if not self.dictionary_path:
            print("Medical dictionary path not specified")
            return False

        try:
            import json
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract terms
            terms = set()
            for item in data:
                if isinstance(item, dict) and "term" in item:
                    terms.add(item["term"])

            self._terms_set = terms
            self._build_trie()
            self._loaded = True

            print(f"Successfully loaded {len(terms)} medical terms")
            return True

        except Exception as e:
            print(f"Failed to load medical dictionary: {e}")
            return False

    def _build_trie(self):
        """Build trie for fast prefix matching."""
        if not self._terms_set:
            return
            
        self._terms_trie = {}
        for term in self._terms_set:
            node = self._terms_trie
            for char in term.lower():
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['$'] = True  # Mark end of term

    def is_medical_term(self, token: str) -> bool:
        """Quick check whether token is a medical term."""
        if not self._loaded:
            return False
        return token.lower() in self._terms_set
    
    def get_medical_terms(self) -> set:
        """Return medical term set."""
        return self._terms_set or set()
    
    def get_performance_stats(self) -> dict:
        """Return performance statistics."""
        if not self._loaded:
            return {"status": "not_loaded"}
            
        term_count = len(self._terms_set)
        return {
            "total_terms": term_count,
            "recommended_method": self._get_recommended_method(term_count),
            "estimated_time": self._estimate_processing_time(term_count),
            "memory_usage": f"{term_count * 20 / 1024 / 1024:.2f} MB"
        }
    
    def _get_recommended_method(self, term_count: int) -> str:
        """Recommend best matching method by term count."""
        if term_count > 10000:
            return "Regex batch matching (recommended)"
        elif term_count > 1000:
            return "Regex batch matching"
        elif term_count > 100:
            return "Optimized substring matching"
        else:
            return "Simple counting method"

    def _estimate_processing_time(self, term_count: int) -> str:
        """Estimate processing time."""
        if term_count > 100000:
            return "> 10s (large dictionary)"
        elif term_count > 10000:
            return "1-5s (medium dictionary)"
        elif term_count > 1000:
            return "0.1-1s (small dictionary)"
        else:
            return "< 0.1s (tiny dictionary)"

class DenoisingQualityGEval:
    """GEval scorer for denoising quality (Accuracy, Integrity, Smoothness)."""
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "deepseek-v3",
        base_url: str = None,
        api_timeout: float = None,
        api_config: dict = None,
    ):
        """Initialize denoising quality scorer."""
        self.api_config = api_config
        if api_key:
            self.client = create_openai_client(
                api_key,
                base_url,
                timeout=api_timeout,
                config=api_config,
            )
            self.model_name = model_name
        else:
            self.client = None
            self.model_name = model_name
    
    def evaluate(self, original_text: str, denoised_text: str) -> Dict[str, float]:
        """
        Score denoised text across multiple dimensions.

        Args:
            original_text: Original text (before detector/editor processing)
            denoised_text: Denoised text

        Returns:
            Dict: Scores for Accuracy, Integrity, and Smoothness
        """
        if not self.client:
            return {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}
        
        system_prompt = PCRAgentPrompts.denoising_quality_system()
        user_prompt = PCRAgentPrompts.denoising_quality_user(original_text, denoised_text)

        default_scores = {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}
        try:
            response = chat_completions_create(
                self.client,
                context="PCRAgent DenoisingQualityGEval denoising quality scoring",
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
                temperature=0.1,
                api_config=self.api_config,
            )
            rating_text = response.choices[0].message.content.strip()
            json_match = re.search(r"\{.*\}", rating_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                result = {}
                for dim in ["accuracy", "integrity", "smoothness"]:
                    result[dim] = float(scores[dim]) if dim in scores else 4.0
                return result
            return default_scores
        except Exception as e:
            print_api_error(
                e,
                context="PCRAgent DenoisingQualityGEval (fallback: default scores)",
                model=self.model_name,
                show_traceback=True,
            )
            return default_scores

class EvaluationMetrics:
    """Evaluation metrics for the PCRAgent framework"""
    
    def __init__(self, medical_terms_manager=None):
        self.medical_terms_manager = medical_terms_manager
        self._regex_patterns_cache = {}  # Cache compiled regex patterns
    
    def calculate_consistency(self, original: str, denoised: str) -> float:
        """Calculate semantic consistency between original and denoised text"""
        # Simple word overlap-based consistency
        original_words = set(original.lower().split())
        denoised_words = set(denoised.lower().split())
        
        if not original_words and not denoised_words:
            return 1.0
        if not original_words or not denoised_words:
            return 0.0
            
        intersection = len(original_words & denoised_words)
        union = len(original_words | denoised_words)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_medical_accuracy(self, text: str, medical_terms: set = None) -> float:
        """Calculate medical accuracy based on preserved medical terms"""
        # Prefer medical dictionary manager
        if self.medical_terms_manager and self.medical_terms_manager._loaded:
            medical_terms = self.medical_terms_manager.get_medical_terms()
        elif not medical_terms:
            return 1.0
            
        if not medical_terms:
            return 1.0
            
        # Use optimized term matching
        preserved_terms = self._count_preserved_medical_terms(text, medical_terms)
        
        return preserved_terms / len(medical_terms) if medical_terms else 1.0
    
    def calculate_medical_term_retention_rate(self, original_text: str, denoised_text: str) -> float:
        """
        Calculate medical term retention rate.

        Args:
            original_text: Original input text
            denoised_text: Final Arbiter output text

        Returns:
            float: Retention rate = medical terms in output / medical terms in original
        """
        # Get medical term set
        if not self.medical_terms_manager or not self.medical_terms_manager._loaded:
            return 1.0  # No dictionary loaded

        medical_terms = self.medical_terms_manager.get_medical_terms()
        if not medical_terms:
            return 1.0

        # Count medical terms in original text
        original_term_count = self._count_preserved_medical_terms(original_text, medical_terms)

        # Count medical terms in denoised text
        denoised_term_count = self._count_preserved_medical_terms(denoised_text, medical_terms)

        # Compute retention rate
        if original_term_count == 0:
            return 1.0  # No medical terms in original

        retention_rate = denoised_term_count / original_term_count
        return retention_rate

    def _count_preserved_medical_terms(self, text: str, medical_terms: set) -> int:
        """Optimized medical term counting."""
        if not medical_terms:
            return 0

        text_lower = text.lower()
        preserved_count = 0

        # Option 1: regex batch matching (recommended for large dictionaries)
        if len(medical_terms) > 1000:
            return self._regex_based_count(text_lower, medical_terms)

        # Option 2: optimized substring matching (medium dictionaries)
        elif len(medical_terms) > 100:
            return self._optimized_substring_count(text_lower, medical_terms)

        # Option 3: simple method (small dictionaries)
        else:
            return self._simple_count(text_lower, medical_terms)

    def _regex_based_count(self, text_lower: str, medical_terms: set) -> int:
        """Regex-based efficient matching (large dictionaries)."""
        import re

        # Cache compiled patterns
        terms_key = frozenset(medical_terms)
        if terms_key not in self._regex_patterns_cache:
            # Build regex pattern
            # Escape special chars; word-boundary matching
            escaped_terms = [re.escape(term.lower()) for term in medical_terms]
            pattern = r'\b(?:' + '|'.join(escaped_terms) + r')\b'
            self._regex_patterns_cache[terms_key] = re.compile(pattern, re.IGNORECASE)

        compiled_pattern = self._regex_patterns_cache[terms_key]

        # Match all terms in one pass
        matches = compiled_pattern.findall(text_lower)

        # Count unique matched terms
        unique_matches = set(matches)
        return len(unique_matches)

    def _optimized_substring_count(self, text_lower: str, medical_terms: set) -> int:
        """Optimized substring matching (medium dictionaries)."""
        preserved_count = 0

        # Sort by length; match longer terms first
        sorted_terms = sorted(medical_terms, key=len, reverse=True)

        for term in sorted_terms:
            if term.lower() in text_lower:
                preserved_count += 1
                # Optional: remove matched term to avoid double counting
                # text_lower = text_lower.replace(term.lower(), '', 1)

        return preserved_count

    def _simple_count(self, text_lower: str, medical_terms: set) -> int:
        """Simple counting method (small dictionaries)."""
        return sum(1 for term in medical_terms if term.lower() in text_lower)

    def load_large_medical_dictionary(self, file_path: str, format: str = "txt") -> set:
        """
        Load large medical dictionary file.

        Args:
            file_path: Dictionary file path
            format: File format ("txt", "csv", "json")

        Returns:
            set: Set of medical terms
        """
        medical_terms = set()
        
        try:
            if format == "txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        term = line.strip()
                        if term:
                            medical_terms.add(term)
            
            elif format == "csv":
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and row[0].strip():
                            medical_terms.add(row[0].strip())
            
            elif format == "json":
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        medical_terms = set(data)
                    elif isinstance(data, dict) and "terms" in data:
                        medical_terms = set(data["terms"])
            
            print(f"Successfully loaded {len(medical_terms)} medical terms")
            return medical_terms

        except Exception as e:
            print(f"Failed to load medical dictionary: {e}")
            return set()

    def get_performance_stats(self, medical_terms: set) -> dict:
        """Return performance statistics."""
        return {
            "total_terms": len(medical_terms),
            "recommended_method": self._get_recommended_method(len(medical_terms)),
            "estimated_time": self._estimate_processing_time(len(medical_terms))
        }

    def _get_recommended_method(self, term_count: int) -> str:
        """Recommend best matching method by term count."""
        if term_count > 10000:
            return "Regex batch matching (recommended)"
        elif term_count > 1000:
            return "Regex batch matching"
        elif term_count > 100:
            return "Optimized substring matching"
        else:
            return "Simple counting method"

    def _estimate_processing_time(self, term_count: int) -> str:
        """Estimate processing time."""
        if term_count > 100000:
            return "> 10s (large dictionary)"
        elif term_count > 10000:
            return "1-5s (medium dictionary)"
        elif term_count > 1000:
            return "0.1-1s (small dictionary)"
        else:
            return "< 0.1s (tiny dictionary)"
    
    
    def calculate_correctness(self, denoised: str, gold_standard: str) -> float:
        """Calculate correctness against gold standard"""
        if not gold_standard:
            return 0.0
            
        # Simple word-level accuracy
        denoised_words = denoised.lower().split()
        gold_words = gold_standard.lower().split()
        
        if not gold_words:
            return 1.0 if not denoised_words else 0.0
            
        # Calculate word-level precision and recall
        denoised_set = set(denoised_words)
        gold_set = set(gold_words)
        
        if not denoised_set and not gold_set:
            return 1.0
        if not denoised_set:
            return 0.0
        if not gold_set:
            return 0.0
            
        precision = len(denoised_set & gold_set) / len(denoised_set)
        recall = len(denoised_set & gold_set) / len(gold_set)
        
        if precision + recall == 0:
            return 0.0
            
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score
    
    def calculate_kappa(self, annotator1_edits: List, annotator2_edits: List) -> float:
        """Calculate Cohen's kappa for inter-annotator agreement"""
        # Simplified kappa calculation for edit agreement
        if not annotator1_edits and not annotator2_edits:
            return 1.0
            
        # Convert edits to comparable format
        def edit_to_key(edit):
            if isinstance(edit, dict):
                return (edit.get('start_char', 0), edit.get('end_char', 0), edit.get('op', ''))
            return (getattr(edit, 'start_char', 0), getattr(edit, 'end_char', 0), getattr(edit, 'op', ''))
        
        edits1_keys = set(edit_to_key(e) for e in annotator1_edits)
        edits2_keys = set(edit_to_key(e) for e in annotator2_edits)
        
        # Calculate agreement
        agreement = len(edits1_keys & edits2_keys)
        total = len(edits1_keys | edits2_keys)
        
        if total == 0:
            return 1.0
            
        observed_agreement = agreement / total
        
        # Expected agreement (simplified)
        expected_agreement = 0.5  # Simplified assumption
        
        if expected_agreement >= 1.0:
            return 1.0
            
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
        return max(0.0, min(1.0, kappa))
    
    def evaluate_all(self, original: str, denoised: str, gold_standard: str = None, 
                    medical_terms: set = None, annotator_edits: List = None) -> Dict:
        """Calculate all evaluation metrics"""
        metrics = {
            "consistency": self.calculate_consistency(original, denoised),
        }
        
        # Use dictionary manager or provided term set
        if self.medical_terms_manager and self.medical_terms_manager._loaded:
            metrics["medical_accuracy"] = self.calculate_medical_accuracy(denoised)
        elif medical_terms:
            metrics["medical_accuracy"] = self.calculate_medical_accuracy(denoised, medical_terms)
        
        if gold_standard:
            metrics["correctness"] = self.calculate_correctness(denoised, gold_standard)
        
        if annotator_edits:
            # Simplified: assume annotator_edits is a list of [annotator1_edits, annotator2_edits]
            if len(annotator_edits) >= 2:
                metrics["kappa"] = self.calculate_kappa(annotator_edits[0], annotator_edits[1])
        
        return metrics

# -------------------------- Main PCRAgent Pipeline --------------------------
class DetectorEditorArbiter:
    """Main PCRAgent (Detector-Editor-Arbiter) Agent for Medical Dialogue Denoising"""
    
    def __init__(
        self,
        medical_dictionary_path: str = None,
        api_key: str = None,
        base_url: str = None,
        api_timeout: float = None,
        api_config: dict = None,
    ):
        """
        Initialize medical dialogue denoising agent.

        Args:
            medical_dictionary_path: Path to medical dictionary file
            api_key: OpenAI API key (optional)
            base_url: API base URL
            api_timeout: Read timeout in seconds (>=180 recommended for long text)
            api_config: Full config dict (includes api_timeout, etc.)
        """
        self.api_config = api_config or {}
        self.api_timeout = api_timeout
        self.api_base_url = base_url or self.api_config.get(
            "base_url", "https://api.chatanywhere.tech/v1"
        )

        # Initialize medical dictionary manager
        self.medical_terms_manager = MedicalTermsManager(medical_dictionary_path)
        if medical_dictionary_path:
            self.medical_terms_manager.load_medical_dictionary()

        llm_kw = dict(
            base_url=self.api_base_url,
            api_timeout=self.api_timeout,
            api_config=self.api_config,
        )

        # Detector modules
        self.gec = GECTagger()
        self.spell = SpellChecker(medical_terms_manager=self.medical_terms_manager)
        self.repetition = RepetitionDetector()
        self.combined_medical = (
            CombinedMedicalDetector(api_key, **llm_kw) if api_key else None
        )

        # Editor pipeline
        self.editor = EditorPipeline(api_key=api_key, **llm_kw)

        # Arbiter pipeline
        self.arbiter = ArbiterPipeline(api_key, **llm_kw)

        # Evaluation
        self.evaluator = EvaluationMetrics(medical_terms_manager=self.medical_terms_manager)

        # Denoising quality scorer (GEval)
        self.quality_evaluator = (
            DenoisingQualityGEval(api_key=api_key, **llm_kw) if api_key else None
        )

        # Quality score thresholds
        self.quality_thresholds = {
            "accuracy": 4.2,
            "integrity": 4.5,
            "smoothness": 3.9
        }
    
    def detect_errors(self, text: str) -> List[SpanEdit]:
        """Detection stage: run all detectors (spell, repetition, grammar, ambiguity, nonmedical)."""
        all_edits = []

        # Run all detectors in fixed order: spell, repetition, grammar, ambiguity, nonmedical
        detectors = [
            ("Spell", self.spell),           # 1. Spell check
            ("Repetition", self.repetition), # 2. Repetition detection
            ("GEC", self.gec),               # 3. Grammar correction
        ]

        if self.combined_medical:
            detectors.append(("CombinedMedical", self.combined_medical))  # 4. Ambiguity (AMB) + 5. Non-medical (NOS)

        for name, detector in detectors:
            try:
                edits = detector.detect(text)
                for edit in edits:
                    edit.detector_name = name
                all_edits.extend(edits)

                # Log AMB and NOS edits separately
                if name == "CombinedMedical":
                    amb_edits = [e for e in edits if e.tag == "AMB"]
                    nos_edits = [e for e in edits if e.tag == "NOS"]
                    if amb_edits:
                        print(f"[AMB] Found {len(amb_edits)} edits")
                    if nos_edits:
                        print(f"[NOS] Found {len(nos_edits)} edits")
                else:
                    print(f"[{name}] Found {len(edits)} edits")
            except Exception as e:
                print(f"[{name}] Error: {e}")
        
        return all_edits
    
    def edit_candidates(self, edits: List[SpanEdit], text: str) -> Dict:
        """Editor stage: process candidate edits."""
        return self.editor.run(edits, text)
    
    def arbitrate_decisions(self, edits: List[SpanEdit], original_text: str, editor_processed_text: str) -> Dict:
        """Arbiter stage: final decisions and conflict resolution."""
        return self.arbiter.run(edits, original_text, editor_processed_text)
    
    def reprocess_with_llm(self, original_text: str, previous_result: str, previous_scores: Dict[str, float], 
                          arbiter_input_text: str) -> str:
        """
        Reprocess text with LLM.

        Args:
            original_text: Original input (before detector/editor)
            previous_result: Output from previous round
            previous_scores: Scores from previous round
            arbiter_input_text: Text fed to Arbiter (after detector/editor)

        Returns:
            str: Reprocessed text
        """
        if not self.arbiter.api_key:
            return previous_result
        
        client = create_openai_client(
            self.arbiter.api_key,
            self.arbiter.base_url,
            timeout=self.arbiter.api_timeout,
            config=self.arbiter.api_config,
        )

        system_prompt = PCRAgentPrompts.reprocess_system()
        user_prompt = PCRAgentPrompts.reprocess_user(
            original_text,
            previous_result,
            previous_scores,
            arbiter_input_text,
            thresholds=getattr(self, "quality_thresholds", {}),
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            out_tokens = estimate_max_tokens(
                original_text,
                previous_result,
                arbiter_input_text,
                system_prompt,
                user_prompt,
                config=self.arbiter.api_config,
            )
            response = chat_completions_create(
                client,
                context="PCRAgent denoising low-score reprocess",
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=out_tokens,
                api_config=self.arbiter.api_config,
            )
            result = response.choices[0].message.content.strip()
            # Parse LLM output (English labels; Chinese kept for legacy prompt compatibility)
            if "Improved text:" in result:
                result = result.split("Improved text:")[-1].strip()
            if "改进后的文本:" in result:
                result = result.split("改进后的文本:")[-1].strip()
            if "Denoised text:" in result:
                result = result.split("Denoised text:")[-1].strip()
            if "去噪后的文本:" in result:
                result = result.split("去噪后的文本:")[-1].strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            return result
        except Exception as e:
            print_api_error(
                e,
                context="PCRAgent denoising low-score reprocess (fallback: previous result)",
                model="gpt-4o-mini",
                show_traceback=True,
            )
            return previous_result
    
    def denoise(self, text: str, gold_standard: str = None, verbose: bool = True) -> Dict:
        """
        Main medical dialogue denoising method with quality scoring and reprocessing.

        Args:
            text: Medical dialogue text to denoise (original system input)
            gold_standard: Gold-standard text (optional, for evaluation)
            verbose: Whether to print detailed progress

        Returns:
            Dict: Denoising results and evaluation metrics
        """
        # Preserve original input (before detector/editor)
        original_input_text = text
        
        if verbose:
            print("=" * 50)
            print("Medical Dialogue Denoising Agent")
            print("=" * 50)
        
        # 1. Detector stage
        if verbose:
            print("\n1. DETECTOR STAGE")
            print("-" * 30)
        detector_edits = self.detect_errors(text)
        if verbose:
            print(f"Total edits detected: {len(detector_edits)}")
        
        # 2. Editor stage  
        if verbose:
            print("\n2. EDITOR STAGE")
            print("-" * 30)
        editor_result = self.edit_candidates(detector_edits, text)
        if verbose:
            print(f"Deterministic edits: {len(editor_result['deterministic_edits'])}")
            print(f"Candidate edits: {len(editor_result['candidate_edits'])}")
        
        # Text passed to Arbiter (after Detector and Editor)
        # editor_result edited_text has deterministic edits applied
        arbiter_input_text = editor_result.get('edited_text', text)
        if not arbiter_input_text:
            arbiter_input_text = text
        
        # 3. Arbiter stage
        if verbose:
            print("\n3. ARBITER STAGE") 
            print("-" * 30)
        all_processed_edits = editor_result['processed_edits']
        arbiter_result = self.arbitrate_decisions(all_processed_edits, original_input_text, arbiter_input_text)
        if verbose:
            print(f"Conflicts detected: {len(arbiter_result['conflicts'])}")
            print(f"Final edits applied: {len(arbiter_result['resolved_edits'])}")
        
        # 4. Denoising quality scoring (after Arbiter produces full sentence)
        if verbose:
            print("\n4. DENOISING QUALITY EVALUATION STAGE")
            print("-" * 30)

        final_text = arbiter_result['final_text']
        all_results = []  # Store results and scores for all rounds

        # Round 1: score Arbiter output
        if self.quality_evaluator:
            quality_scores = self.quality_evaluator.evaluate(original_input_text, final_text)
        else:
            quality_scores = {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}
        
        all_results.append({
            "result": final_text,
            "scores": quality_scores.copy(),
            "round": 1
        })
        
        if verbose:
            print(f"Quality Scores (Round 1):")
            print(f"  Accuracy: {quality_scores.get('accuracy', 0):.2f} (threshold: {self.quality_thresholds['accuracy']})")
            print(f"  Integrity: {quality_scores.get('integrity', 0):.2f} (threshold: {self.quality_thresholds['integrity']})")
            print(f"  Smoothness: {quality_scores.get('smoothness', 0):.2f} (threshold: {self.quality_thresholds['smoothness']})")
        
        # Check whether thresholds are met
        meets_threshold = (
            quality_scores.get('accuracy', 0) >= self.quality_thresholds['accuracy'] and
            quality_scores.get('integrity', 0) >= self.quality_thresholds['integrity'] and
            quality_scores.get('smoothness', 0) >= self.quality_thresholds['smoothness']
        )
        
        # Reprocess if below threshold (max 3 rounds)
        max_rounds = 3
        current_round = 1

        while not meets_threshold and current_round < max_rounds:
            current_round += 1
            if verbose:
                print(f"\nReprocessing (Round {current_round})...")

            # Reprocess with LLM
            reprocessed_text = self.reprocess_with_llm(
                original_text=original_input_text,
                previous_result=final_text,
                previous_scores=quality_scores,
                arbiter_input_text=arbiter_input_text
            )
            
            # Score reprocessed result
            if self.quality_evaluator:
                quality_scores = self.quality_evaluator.evaluate(original_input_text, reprocessed_text)
            else:
                quality_scores = {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}
            
            all_results.append({
                "result": reprocessed_text,
                "scores": quality_scores.copy(),
                "round": current_round
            })
            
            if verbose:
                print(f"Quality Scores (Round {current_round}):")
                print(f"  Accuracy: {quality_scores.get('accuracy', 0):.2f} (threshold: {self.quality_thresholds['accuracy']})")
                print(f"  Integrity: {quality_scores.get('integrity', 0):.2f} (threshold: {self.quality_thresholds['integrity']})")
                print(f"  Smoothness: {quality_scores.get('smoothness', 0):.2f} (threshold: {self.quality_thresholds['smoothness']})")
            
            # Update final_text
            final_text = reprocessed_text

            # Re-check thresholds
            meets_threshold = (
                quality_scores.get('accuracy', 0) >= self.quality_thresholds['accuracy'] and
                quality_scores.get('integrity', 0) >= self.quality_thresholds['integrity'] and
                quality_scores.get('smoothness', 0) >= self.quality_thresholds['smoothness']
            )
        
        # After 3 rounds, pick best result if still below threshold
        if not meets_threshold and len(all_results) >= 3:
            if verbose:
                print(f"\nThreshold not met after 3 rounds; selecting best result...")

            # Pick highest accuracy, then integrity, then smoothness
            best_result = all_results[0]
            for result in all_results[1:]:
                current_scores = result["scores"]
                best_scores = best_result["scores"]

                # Compare accuracy first
                if current_scores.get('accuracy', 0) > best_scores.get('accuracy', 0):
                    best_result = result
                elif current_scores.get('accuracy', 0) == best_scores.get('accuracy', 0):
                    # Tie on accuracy: compare integrity
                    if current_scores.get('integrity', 0) > best_scores.get('integrity', 0):
                        best_result = result
                    elif current_scores.get('integrity', 0) == best_scores.get('integrity', 0):
                        # Tie on integrity: compare smoothness
                        if current_scores.get('smoothness', 0) > best_scores.get('smoothness', 0):
                            best_result = result

            final_text = best_result["result"]
            quality_scores = best_result["scores"]

            if verbose:
                print(f"Selected Round {best_result['round']} result (best scores)")

        # 5. Medical term retention rate (when thresholds met or after 3 rounds)
        medical_term_retention_rate = None
        if meets_threshold or len(all_results) >= 3:
            # Compute retention rate
            medical_term_retention_rate = self.evaluator.calculate_medical_term_retention_rate(
                original_input_text, final_text
            )
            if verbose:
                print(f"\nMedical term retention rate: {medical_term_retention_rate:.3f}")
                if self.evaluator.medical_terms_manager and self.evaluator.medical_terms_manager._loaded:
                    medical_terms = self.evaluator.medical_terms_manager.get_medical_terms()
                    original_count = self.evaluator._count_preserved_medical_terms(original_input_text, medical_terms)
                    denoised_count = self.evaluator._count_preserved_medical_terms(final_text, medical_terms)
                    print(f"  Medical terms in original: {original_count}")
                    print(f"  Medical terms after denoising: {denoised_count}")
                else:
                    print("  Medical term dictionary not loaded")

        # 6. Traditional evaluation (legacy metrics)
        if verbose:
            print("\n6. TRADITIONAL EVALUATION STAGE")
            print("-" * 30)
        evaluation = self.evaluator.evaluate_all(
            original=text,
            denoised=final_text,
            gold_standard=gold_standard
        )
        
        if verbose:
            print("Evaluation Metrics:")
            for metric, value in evaluation.items():
                print(f"  {metric}: {value:.3f}")

        return {
            "original_text": text,
            "detector_edits": detector_edits,
            "editor_result": editor_result,
            "arbiter_result": arbiter_result,
            "final_text": final_text,
            "evaluation": evaluation,
            "quality_scores": quality_scores,  # Denoising quality scores
            "quality_evaluation_rounds": len(all_results),  # Number of scoring rounds
            "all_quality_results": all_results,  # All round results
            "medical_term_retention_rate": medical_term_retention_rate  # Medical term retention rate
        }

    def quick_denoise(self, text: str) -> str:
        """
        Quick denoise: return denoised text only.

        Args:
            text: Medical dialogue text to denoise

        Returns:
            str: Denoised text
        """
        result = self.denoise(text, verbose=False)
        return result['final_text']

    def batch_denoise(self, texts: List[str], verbose: bool = True) -> List[Dict]:
        """
        Batch denoise multiple texts.

        Args:
            texts: List of texts to denoise
            verbose: Whether to print progress

        Returns:
            List[Dict]: Denoising result per text
        """
        results = []
        for i, text in enumerate(texts):
            if verbose:
                print(f"\nProcessing text {i+1}/{len(texts)}...")
            result = self.denoise(text, verbose=False)
            results.append(result)
        return results

# Backward-compatible alias
PCRAgent = DetectorEditorArbiter

# -------------------------- Configuration notes --------------------------
"""
Parameters to configure before use:

1. Model path: local GECTagger grammar-correction model
2. API key: OpenAI-compatible endpoint for CombinedMedical / Editor / Arbiter / GEval
3. Medical dictionary: medical_terms.json
4. LLM prompts: edit PCRAgentPrompts class in pcr_agent_prompt.py
"""


# Export main classes for external use
__all__ = [
    'DetectorEditorArbiter',
    'PCRAgent',
    'PCRAgentPrompts',
    'SpanEdit',
    'BaseExtractorModule',
    'GECTagger',
    'SpellChecker', 
    'RepetitionDetector',
    'CombinedMedicalDetector',
    'EditorPipeline',
    'ArbiterPipeline',
    'EvaluationMetrics',
    'MedicalTermsManager'
]
