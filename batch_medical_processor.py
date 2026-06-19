"""
Pre-consultation batch processor.

Workflow overview:
1. Denoise input text via medical_denoising_agent (unchanged)
2. Multi-turn pre-consultation staff guidance (HPI / past history / allergy / family history)
3. Generate pre-consultation report (including conversation summary)
4. Research evaluation: holistic dialogue-reply scoring + report scoring (optional, not core service)

CSV batch: first column is raw noisy input; output includes pre-consultation report and
per-dimension score columns (model replies / report × accuracy, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from llm_api_utils import (
    chat_completions_create,
    create_openai_client,
    estimate_max_tokens,
    print_api_error,
)
from medical_denoising_agent import DetectorEditorArbiter
from pre_consultation_prompts import (
    COMPLETION_QUESTION,
    CONFIRM_END_PROMPT,
    PreConsultationPrompts,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "config.json")
DEFAULT_MEDICAL_DICT = os.path.join(_SCRIPT_DIR, "medical_terms.json")
DEFAULT_REPORT_METRICS = os.path.join(_SCRIPT_DIR, "Report_Evaluation_Metrics.json")

HISTORY_MODULE_KEYS = [
    "present_illness",
    "past_history",
    "allergy_history",
    "family_history",
]

# Pre-consultation dialogue tracks four histories; chief complaint / personal history extracted at report time
REPORT_MODULE_KEYS = [
    "chief_complaint",
    "present_illness",
    "past_history",
    "allergy_history",
    "family_history",
    "personal_history",
]

MODULE_LABELS = {
    "present_illness": "History of present illness (HPI)",
    "past_history": "Past medical history",
    "allergy_history": "Allergy history",
    "family_history": "Family history",
}

REPORT_MODULE_LABELS = {
    "chief_complaint": "Chief complaint",
    "present_illness": "History of present illness (HPI)",
    "past_history": "Past medical history",
    "allergy_history": "Allergy history",
    "family_history": "Family history",
    "personal_history": "Personal / social history",
}

END_INTENT_PATTERNS = re.compile(
    r"(结束|完了|没有了|就这些|可以了|不用了|没了|到此为止|预问诊结束|"
    r"end\b|that's all|nothing else|all good|no more|done\b)",
    re.IGNORECASE,
)

RESEARCH_SCORE_DIMENSIONS = ["Accuracy", "Completeness", "Security", "Clarity"]

SCORE_DIMENSION_LABELS = {
    "Accuracy": "Accuracy",
    "Completeness": "Completeness",
    "Security": "Security",
    "Clarity": "Clarity",
}


def build_output_csv_columns() -> List[str]:
    """Build batch output CSV columns: base columns + one column per score dimension."""
    cols = ["Original input", "Denoised text", "Pre-consultation report"]
    for dim in RESEARCH_SCORE_DIMENSIONS:
        label = SCORE_DIMENSION_LABELS[dim]
        cols.append(f"Model reply_{label}")
    for dim in RESEARCH_SCORE_DIMENSIONS:
        label = SCORE_DIMENSION_LABELS[dim]
        cols.append(f"Report_{label}")
    return cols


OUTPUT_CSV_COLUMNS = build_output_csv_columns()


class SessionPhase(str, Enum):
    COLLECTING = "collecting"
    AWAITING_COMPLETION_ACK = "awaiting_completion_ack"
    AWAITING_END_CONFIRM = "awaiting_end_confirm"
    FINISHED = "finished"


@dataclass
class HistoryModules:
    """History modules; None = not yet collected; 'none' etc. = patient explicitly denied."""

    chief_complaint: Optional[str] = None
    present_illness: Optional[str] = None
    past_history: Optional[str] = None
    allergy_history: Optional[str] = None
    family_history: Optional[str] = None
    personal_history: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "chief_complaint": self.chief_complaint,
            "present_illness": self.present_illness,
            "past_history": self.past_history,
            "allergy_history": self.allergy_history,
            "family_history": self.family_history,
            "personal_history": self.personal_history,
        }

    def all_filled(self) -> bool:
        return all(v is not None and str(v).strip() != "" for v in self.as_dict().values())

    def missing_labels(self) -> List[str]:
        missing = []
        for key, label in MODULE_LABELS.items():
            val = getattr(self, key)
            if val is None or str(val).strip() == "":
                missing.append(label)
        return missing


@dataclass
class PreConsultationReport:
    modules: HistoryModules
    conversation_summary: str = ""

    def to_text(self) -> str:
        lines = ["=" * 40, "Pre-Consultation Report", "=" * 40]
        for key in REPORT_MODULE_KEYS:
            label = REPORT_MODULE_LABELS[key]
            value = getattr(self.modules, key) or "(Not collected)"
            lines.append(f"\n[{label}]\n{value}")
        lines.append(f"\n[Conversation Summary]\n{self.conversation_summary}")
        lines.append("=" * 40)
        return "\n".join(lines)


@dataclass
class PreConsultationSession:
    """State for a single pre-consultation session."""

    messages: List[Dict[str, str]] = field(default_factory=list)
    modules: HistoryModules = field(default_factory=HistoryModules)
    phase: SessionPhase = SessionPhase.COLLECTING
    completion_question_asked: bool = False
    inner_world_notes: List[str] = field(default_factory=list)
    staff_replies: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if role == "assistant":
            self.staff_replies.append(content)

    def transcript(self) -> str:
        parts = []
        for m in self.messages:
            speaker = "Pre-consultation staff" if m["role"] == "assistant" else "Patient"
            parts.append(f"{speaker}: {m['content']}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Batch pre-consultation processor
# ---------------------------------------------------------------------------


class BatchMedicalProcessor:
    """Pre-consultation batch processor: denoise → multi-turn pre-consultation → report → research scoring."""

    def __init__(self, config_file: Optional[str] = None):
        self.config = self._default_config()
        if config_file:
            self.load_config(config_file)
        elif os.path.isfile(DEFAULT_CONFIG_PATH):
            self.load_config(DEFAULT_CONFIG_PATH)

        self.client = create_openai_client(
            self.config["api_key"],
            self.config.get("base_url"),
            config=self.config,
        )
        self.chat_model = self.config.get("chat_model", "deepseek-v3")
        print(
            f"LLM endpoint: {self.config.get('base_url', 'https://api.chatanywhere.tech/v1')} "
            f"| model: {self.chat_model} "
            f"| read timeout: {self.config.get('api_timeout', 180)}s"
        )

        self.report_metrics = self._load_metrics(
            self.config.get("report_evaluation_metrics_path", DEFAULT_REPORT_METRICS)
        )

        self.denoiser: Optional[DetectorEditorArbiter] = None
        if self.config.get("enable_denoising", True):
            self._init_denoiser()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "api_key": "",
            "base_url": "https://api.chatanywhere.tech/v1",
            "chat_model": "deepseek-v3",
            "verbose": False,
            "enable_denoising": True,
            "medical_dictionary_path": DEFAULT_MEDICAL_DICT,
            "report_evaluation_metrics_path": DEFAULT_REPORT_METRICS,
            "max_dialogue_turns": 24,
            "enable_research_scoring": True,
            "api_timeout": 180,
            "api_timeout_connect": 30,
            "llm_max_output_tokens": 16384,
            "llm_output_token_multiplier": 2.5,
        }

    def load_config(self, config_file: str) -> None:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                self.config.update(json.load(f))
        except Exception as e:
            print(f"Failed to load config file: {e}")

    def _init_denoiser(self) -> None:
        dict_path = self.config.get("medical_dictionary_path", DEFAULT_MEDICAL_DICT)
        print("Initializing medical dialogue denoiser...")
        try:
            self.denoiser = DetectorEditorArbiter(
                medical_dictionary_path=dict_path,
                api_key=self.config["api_key"],
                base_url=self.config.get("base_url"),
                api_timeout=self.config.get("api_timeout"),
                api_config=self.config,
            )
            print("Medical dialogue denoiser initialized successfully")
        except Exception as e:
            print(f"Denoiser initialization failed: {e}")
            self.denoiser = None

    def _load_metrics(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: v for k, v in data.items() if k in RESEARCH_SCORE_DIMENSIONS}
        except Exception as e:
            print(f"Failed to load evaluation metrics ({path}): {e}")
            return {}

    # ------------------------- LLM utilities -------------------------

    def _chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
        context: str = "Pre-consultation LLM",
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if max_tokens is None or max_tokens <= 0:
            max_tokens = estimate_max_tokens(system, user, config=self.config)
        try:
            response = chat_completions_create(
                self.client,
                context=context,
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_config=self.config,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print_api_error(
                e,
                context=context,
                model=self.chat_model,
                base_url=self.config.get("base_url"),
                show_traceback=True,
            )
            raise

    @staticmethod
    def _parse_json_block(text: str) -> Optional[Dict]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    def denoise_text(self, text: str) -> Tuple[str, Optional[str]]:
        if not self.config.get("enable_denoising", True) or not self.denoiser:
            return text, None
        try:
            result = self.denoiser.denoise(text, verbose=False)
            return result["final_text"], None
        except Exception as e:
            print_api_error(
                e,
                context="Medical dialogue denoising (DEA)",
                base_url=self.config.get("base_url"),
                show_traceback=True,
            )
            return text, str(e)

    # ------------------------- Input type classification -------------------------

    def classify_input(self, text: str) -> str:
        """
        Return 'complete_dialogue' or 'patient_utterance'.
        """
        system = PreConsultationPrompts.classify_input_system()
        user = PreConsultationPrompts.classify_input_user(text)
        raw = self._chat(
            system, user, temperature=0.0, max_tokens=80, context="Input type classification"
        )
        data = self._parse_json_block(raw) or {}
        t = data.get("type", "patient_utterance")
        if t not in ("complete_dialogue", "patient_utterance"):
            # Heuristic: multi-turn physician–patient markers
            if len(text) > 200 and (
                "医生" in text or "医师" in text or "Doctor:" in text or "Patient:" in text
            ):
                return "complete_dialogue"
            return "patient_utterance"
        return t

    # ------------------------- History module extraction / update -------------------------

    def extract_modules_from_text(self, text: str, *, is_dialogue: bool) -> HistoryModules:
        system = PreConsultationPrompts.extract_modules_system()
        user = PreConsultationPrompts.extract_modules_user(text, is_dialogue=is_dialogue)
        raw = self._chat(system, user, temperature=0.1, max_tokens=None, context="History extraction")
        data = self._parse_json_block(raw) or {}

        def norm(v: Any) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            if s.lower() in ("null", "none", "未提及", "not mentioned", ""):
                return None
            return s

        modules = HistoryModules(
            chief_complaint=norm(data.get("chief_complaint")),
            present_illness=norm(data.get("present_illness")),
            past_history=norm(data.get("past_history")),
            allergy_history=norm(data.get("allergy_history")),
            family_history=norm(data.get("family_history")),
            personal_history=norm(data.get("personal_history")),
        )
        return self._finalize_history_modules(modules, text)

    def _finalize_history_modules(
        self, modules: HistoryModules, source_text: str
    ) -> HistoryModules:
        """Programmatic deduplication / sectioning + optional LLM polish."""
        from clinical_report_polish import polish_history_modules

        polish_history_modules(modules)
        try:
            modules = self.refine_modules_with_llm(modules, source_text)
        except Exception as e:
            print(f"Skipping LLM history polish: {e}")
        polish_history_modules(modules)
        return modules

    def refine_modules_with_llm(
        self, modules: HistoryModules, source_text: str
    ) -> HistoryModules:
        import json

        draft = {
            k: getattr(modules, k)
            for k in REPORT_MODULE_KEYS
        }
        system = PreConsultationPrompts.refine_modules_system()
        user = PreConsultationPrompts.refine_modules_user(
            json.dumps(draft, ensure_ascii=False), source_text
        )
        raw = self._chat(
            system,
            user,
            temperature=0.12,
            max_tokens=2200,
            context="History polish",
        )
        data = self._parse_json_block(raw) or {}

        def norm(v: Any) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            if s.lower() in ("null", "none", "未提及", "not mentioned", ""):
                return None
            return s

        return HistoryModules(
            chief_complaint=norm(data.get("chief_complaint")) or modules.chief_complaint,
            present_illness=norm(data.get("present_illness")) or modules.present_illness,
            past_history=norm(data.get("past_history")) or modules.past_history,
            allergy_history=norm(data.get("allergy_history")) or modules.allergy_history,
            family_history=norm(data.get("family_history")) or modules.family_history,
            personal_history=norm(data.get("personal_history")) or modules.personal_history,
        )

    def merge_modules(self, current: HistoryModules, incoming: HistoryModules) -> None:
        for key in REPORT_MODULE_KEYS:
            new_val = getattr(incoming, key)
            if new_val is None or str(new_val).strip() == "":
                continue
            old_val = getattr(current, key)
            if old_val is None or str(old_val).strip() == "":
                setattr(current, key, new_val)
            else:
                if new_val.strip() != old_val.strip() and new_val.strip() not in ("无", "none", "None"):
                    setattr(current, key, f"{old_val}; {new_val}")

    # ------------------------- Pre-consultation dialogue core -------------------------

    def reconstruct_inner_world(self, patient_text: str, session: PreConsultationSession) -> str:
        """Reconstruct first-person inner perspective (for staff tone adjustment; not shown to patient)."""
        system = PreConsultationPrompts.inner_world_system()
        ctx = session.transcript() if session.messages else ""
        user = PreConsultationPrompts.inner_world_user(ctx, patient_text)
        note = self._chat(
            system, user, temperature=0.45, max_tokens=220, context="Patient inner perspective"
        )
        session.inner_world_notes.append(note)
        return note

    def generate_staff_reply(
        self,
        session: PreConsultationSession,
        inner_world: str,
        *,
        force_completion_question: bool = False,
        force_confirm_prompt: bool = False,
    ) -> str:
        modules_status = "\n".join(
            f"- {MODULE_LABELS[k]}: {getattr(session.modules, k) or '(Pending)'}"
            for k in HISTORY_MODULE_KEYS
        )
        missing = session.modules.missing_labels()
        if force_confirm_prompt:
            phase_instruction = PreConsultationPrompts.staff_phase_force_confirm()
        elif force_completion_question:
            phase_instruction = PreConsultationPrompts.staff_phase_force_completion()
        else:
            phase_instruction = PreConsultationPrompts.staff_phase_collecting(missing)

        user = PreConsultationPrompts.staff_user(
            inner_world=inner_world,
            modules_status=modules_status,
            missing_labels=", ".join(missing) if missing else "",
            transcript=session.transcript(),
            phase_instruction=phase_instruction,
        )

        return self._chat(
            PreConsultationPrompts.staff_system(),
            user,
            temperature=0.5,
            max_tokens=520,
            context="Pre-consultation staff reply",
        )

    def detect_end_intent(self, text: str) -> bool:
        if END_INTENT_PATTERNS.search(text.strip()):
            return True
        system = PreConsultationPrompts.end_intent_system()
        user = PreConsultationPrompts.end_intent_user(text)
        raw = self._chat(
            system, user, temperature=0.0, max_tokens=60, context="End-intent detection"
        )
        data = self._parse_json_block(raw) or {}
        return bool(data.get("end", False))

    def parse_confirm_yn(self, text: str) -> Optional[bool]:
        t = text.strip().upper()
        if t in ("Y", "YES", "是", "确认", "好", "确定", "CONFIRM", "OK"):
            return True
        if t in ("N", "NO", "否", "不", "继续", "CONTINUE"):
            return False
        return None

    def simulate_patient_reply(
        self,
        staff_message: str,
        initial_context: str,
        session: PreConsultationSession,
    ) -> str:
        """In batch mode, LLM simulates patient replies from initial context and dialogue history."""
        system = PreConsultationPrompts.simulate_patient_system()
        user = PreConsultationPrompts.simulate_patient_user(
            initial_context,
            session.transcript(),
            staff_message,
        )
        return self._chat(
            system, user, temperature=0.55, max_tokens=300, context="Simulated patient reply"
        )

    def run_pre_consultation_dialogue(
        self,
        denoised_text: str,
        *,
        interactive: bool = False,
        initial_patient_text: Optional[str] = None,
    ) -> PreConsultationSession:
        session = PreConsultationSession()
        patient_seed = initial_patient_text or denoised_text
        max_turns = int(self.config.get("max_dialogue_turns", 24))

        # First patient message
        session.add_message("user", patient_seed)
        self.merge_modules(session.modules, self.extract_modules_from_text(patient_seed, is_dialogue=False))

        for turn in range(max_turns):
            last_patient = session.messages[-1]["content"]

            if session.phase == SessionPhase.AWAITING_END_CONFIRM:
                confirmed = self.parse_confirm_yn(last_patient)
                if confirmed is True:
                    session.phase = SessionPhase.FINISHED
                    break
                if confirmed is False:
                    session.phase = SessionPhase.COLLECTING
                    session.completion_question_asked = True
                else:
                    session.add_message(
                        "assistant",
                        "Sorry, I did not understand your choice. If pre-consultation information is "
                        "complete and you wish to end, enter Y; if you have more to add, enter N.",
                    )
                    if interactive:
                        reply = input("Patient> ").strip()
                    else:
                        reply = "Y"
                    session.add_message("user", reply)
                    continue

            inner_world = self.reconstruct_inner_world(last_patient, session)

            force_completion = False
            force_confirm = False

            if session.phase == SessionPhase.AWAITING_COMPLETION_ACK:
                if self.detect_end_intent(last_patient):
                    session.phase = SessionPhase.AWAITING_END_CONFIRM
                    force_confirm = True
                # Otherwise continue collecting history
            elif session.modules.all_filled() and not session.completion_question_asked:
                session.phase = SessionPhase.AWAITING_COMPLETION_ACK
                session.completion_question_asked = True
                force_completion = True

            staff_reply = self.generate_staff_reply(
                session,
                inner_world,
                force_completion_question=force_completion,
                force_confirm_prompt=force_confirm,
            )
            session.add_message("assistant", staff_reply)

            if session.phase == SessionPhase.FINISHED:
                break

            if session.phase == SessionPhase.AWAITING_END_CONFIRM:
                if interactive:
                    patient_reply = input("Patient> ").strip()
                else:
                    patient_reply = "Y"
                session.add_message("user", patient_reply)
                confirmed = self.parse_confirm_yn(patient_reply)
                if confirmed is True:
                    session.phase = SessionPhase.FINISHED
                    break
                session.phase = SessionPhase.COLLECTING
                continue

            if interactive:
                patient_reply = input("Patient> ").strip()
            else:
                patient_reply = self.simulate_patient_reply(
                    staff_reply, denoised_text, session
                )

            session.add_message("user", patient_reply)
            self.merge_modules(
                session.modules,
                self.extract_modules_from_text(patient_reply, is_dialogue=False),
            )

            if session.phase == SessionPhase.AWAITING_COMPLETION_ACK and self.detect_end_intent(
                patient_reply
            ):
                session.phase = SessionPhase.AWAITING_END_CONFIRM
                confirm_msg = self.generate_staff_reply(
                    session,
                    inner_world,
                    force_confirm_prompt=True,
                )
                session.add_message("assistant", confirm_msg)
                if interactive:
                    yn = input("Patient> ").strip()
                else:
                    yn = "Y"
                session.add_message("user", yn)
                if self.parse_confirm_yn(yn) is not False:
                    session.phase = SessionPhase.FINISHED
                    break

        if session.phase != SessionPhase.FINISHED:
            session.phase = SessionPhase.FINISHED
        return session

    def run_complete_dialogue_path(self, denoised_text: str, *, interactive: bool = False) -> PreConsultationSession:
        """Input is already a complete physician–patient dialogue: extract modules → completeness confirmation."""
        session = PreConsultationSession()
        session.modules = self.extract_modules_from_text(denoised_text, is_dialogue=True)

        # Add dialogue as context (simplified: full passage as patient + staff context)
        session.add_message("user", denoised_text)
        session.add_message("assistant", COMPLETION_QUESTION)
        session.completion_question_asked = True
        session.phase = SessionPhase.AWAITING_COMPLETION_ACK

        if interactive:
            patient_reply = input("Patient> ").strip() or "end"
        else:
            patient_reply = "end"

        session.add_message("user", patient_reply)

        if self.detect_end_intent(patient_reply):
            session.add_message("assistant", CONFIRM_END_PROMPT)
            session.phase = SessionPhase.AWAITING_END_CONFIRM
            if interactive:
                yn = input("Patient> ").strip()
            else:
                yn = "Y"
            session.add_message("user", yn)
        else:
            # Re-extract after supplemental information
            self.merge_modules(
                session.modules,
                self.extract_modules_from_text(patient_reply, is_dialogue=False),
            )
            session.add_message("assistant", COMPLETION_QUESTION)
            if interactive:
                patient_reply2 = input("Patient> ").strip() or "end"
            else:
                patient_reply2 = "end"
            session.add_message("user", patient_reply2)
            session.add_message("assistant", CONFIRM_END_PROMPT)
            if interactive:
                yn = input("Patient> ").strip()
            else:
                yn = "Y"
            session.add_message("user", yn)

        session.phase = SessionPhase.FINISHED
        return session

    @staticmethod
    def _normalize_conversation_summary(text: str) -> str:
        t = (text or "").strip()
        for prefix in (
            "Conversation Summary",
            "conversation summary",
            "【对话总结】",
            "对话总结：",
            "对话总结:",
            "对话总结",
        ):
            if t.lower().startswith(prefix.lower()):
                t = t[len(prefix) :].lstrip(" \n:：-")
        return t.strip()

    def generate_conversation_summary(self, session: PreConsultationSession) -> str:
        system = PreConsultationPrompts.conversation_summary_system()
        user = PreConsultationPrompts.conversation_summary_user(session.transcript())
        raw = self._chat(
            system, user, temperature=0.25, max_tokens=520, context="Conversation summary"
        )
        return self._normalize_conversation_summary(raw)

    def build_report(self, session: PreConsultationSession) -> PreConsultationReport:
        # Final full-transcript extraction to ensure modules are complete
        transcript = session.transcript()
        final_modules = self.extract_modules_from_text(transcript, is_dialogue=True)
        self.merge_modules(session.modules, final_modules)
        session.modules = self._finalize_history_modules(session.modules, transcript)
        summary = self.generate_conversation_summary(session)
        return PreConsultationReport(modules=session.modules, conversation_summary=summary)

    # ------------------------- Research scoring (non-core service) -------------------------

    def score_dialogue_replies(self, session: PreConsultationSession) -> Dict[str, float]:
        if not self.report_metrics:
            return {d: 3.0 for d in RESEARCH_SCORE_DIMENSIONS}
        staff_block = "\n---\n".join(session.staff_replies) or "(No staff replies)"
        prompt = PreConsultationPrompts.research_rating_user(
            title="[Task] Score the pre-consultation staff's overall reply quality across this session (do not score individual utterances).",
            content=(
                f"=== Full conversation log ===\n{session.transcript()}\n\n"
                f"=== All staff replies (combined) ===\n{staff_block}"
            ),
            metrics=self.report_metrics,
        )
        raw = self._chat(
            PreConsultationPrompts.research_dialogue_score_system(),
            prompt,
            temperature=0.15,
            max_tokens=220,
            context="Research dialogue-reply scoring",
        )
        data = self._parse_json_block(raw) or {}
        return self._normalize_scores(data)

    def score_report(self, report: PreConsultationReport) -> Dict[str, float]:
        if not self.report_metrics:
            return {d: 3.0 for d in RESEARCH_SCORE_DIMENSIONS}
        prompt = PreConsultationPrompts.research_rating_user(
            title="[Task] Score the quality of the following pre-consultation report.",
            content=report.to_text(),
            metrics=self.report_metrics,
        )
        raw = self._chat(
            PreConsultationPrompts.research_report_score_system(),
            prompt,
            temperature=0.15,
            max_tokens=220,
            context="Research report scoring",
        )
        data = self._parse_json_block(raw) or {}
        return self._normalize_scores(data)

    @staticmethod
    def _normalize_scores(data: Dict) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for dim in RESEARCH_SCORE_DIMENSIONS:
            val = data.get(dim, data.get(dim.lower(), 3))
            try:
                out[dim] = float(val)
            except (TypeError, ValueError):
                out[dim] = 3.0
        return out

    @staticmethod
    def scores_to_json(scores: Dict[str, float]) -> str:
        return json.dumps(scores, ensure_ascii=False)

    @staticmethod
    def scores_to_csv_fields(
        scores: Optional[Dict[str, float]], prefix: str
    ) -> Dict[str, Any]:
        """Expand score dict into CSV columns (one column per dimension)."""
        fields: Dict[str, Any] = {}
        for dim in RESEARCH_SCORE_DIMENSIONS:
            col = f"{prefix}_{SCORE_DIMENSION_LABELS[dim]}"
            if scores is not None and dim in scores:
                fields[col] = scores[dim]
            else:
                fields[col] = ""
        return fields

    @staticmethod
    def build_csv_row(
        *,
        original_input: str,
        denoised_text: str,
        report_text: str,
        dialogue_score: Optional[Dict[str, float]] = None,
        report_score: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        row = {
            "Original input": original_input,
            "Denoised text": denoised_text,
            "Pre-consultation report": report_text,
        }
        row.update(BatchMedicalProcessor.scores_to_csv_fields(dialogue_score, "Model reply"))
        row.update(BatchMedicalProcessor.scores_to_csv_fields(report_score, "Report"))
        return row

    @staticmethod
    def detect_csv_encoding(csv_path: str) -> str:
        """Detect CSV text encoding; prefer UTF-8; support GBK/GB18030 on Chinese Windows."""
        with open(csv_path, "rb") as f:
            sample = f.read(65536)
        if sample.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        for enc in ("utf-8", "gb18030", "gbk"):
            try:
                sample.decode(enc)
                return enc
            except UnicodeDecodeError:
                continue
        return "utf-8-sig"

    @staticmethod
    def _save_output_csv(records: List[Dict[str, Any]], output_csv: str) -> None:
        """Write CSV in fixed column order (utf-8-sig for Excel compatibility)."""
        df = pd.DataFrame(records)
        for col in OUTPUT_CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[OUTPUT_CSV_COLUMNS]
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # ------------------------- Single / batch processing -------------------------

    def process_single(
        self,
        raw_text: str,
        *,
        interactive: bool = False,
        run_research_scoring: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if run_research_scoring is None:
            run_research_scoring = bool(self.config.get("enable_research_scoring", True))

        denoised, denoise_err = self.denoise_text(raw_text)
        input_type = self.classify_input(denoised)

        if input_type == "complete_dialogue":
            session = self.run_complete_dialogue_path(denoised, interactive=interactive)
        else:
            session = self.run_pre_consultation_dialogue(
                denoised, interactive=interactive, initial_patient_text=denoised
            )

        report = self.build_report(session)

        result: Dict[str, Any] = {
            "original_input": raw_text,
            "denoised_text": denoised,
            "denoise_error": denoise_err,
            "input_type": input_type,
            "report_text": report.to_text(),
            "report": report,
            "session": session,
            "dialogue_score": None,
            "report_score": None,
        }

        if run_research_scoring:
            result["dialogue_score"] = self.score_dialogue_replies(session)
            result["report_score"] = self.score_report(report)

        return result

    def process_csv(
        self,
        input_csv: str,
        output_csv: str,
        *,
        has_header: bool = False,
        encoding: str = "auto",
        interactive: bool = False,
        start_row: int = 0,
        end_row: Optional[int] = None,
    ) -> None:
        if encoding == "auto":
            encoding = self.detect_csv_encoding(input_csv)
        df = pd.read_csv(
            input_csv,
            header=0 if has_header else None,
            encoding=encoding,
        )
        if df.shape[1] < 1:
            raise ValueError("Input CSV must have at least one column (raw noisy text)")

        rows = df.iloc[:, 0].astype(str).tolist()
        n = len(rows)
        end = n if end_row is None else min(end_row, n)
        total = max(0, end - start_row)

        print(f"Input file: {input_csv}")
        print(f"Detected encoding: {encoding} (GBK/UTF-8 supported)")
        print(f"Total rows: {n} (processing rows {start_row}–{end - 1}, count {total})")
        print(f"No-header mode: {'No (first row is header)' if has_header else 'Yes (first row is data)'}")
        print("-" * 50)

        out_records = []
        progress = tqdm(
            range(start_row, end),
            desc="Pre-consultation batch processing",
            unit="row",
            total=total,
            dynamic_ncols=True,
        )
        for i in progress:
            progress.set_postfix_str(f"Current row {i + 1}/{n}", refresh=True)
            raw = rows[i].strip()
            if not raw or raw.lower() == "nan":
                continue
            try:
                item = self.process_single(raw, interactive=interactive)
                out_records.append(
                    self.build_csv_row(
                        original_input=item["original_input"],
                        denoised_text=item["denoised_text"],
                        report_text=item.get("report_text", ""),
                        dialogue_score=item.get("dialogue_score"),
                        report_score=item.get("report_score"),
                    )
                )
            except Exception as e:
                print_api_error(
                    e,
                    context=f"Batch processing row {i}",
                    model=self.chat_model,
                    base_url=self.config.get("base_url"),
                    show_traceback=True,
                )
                print(f"Row {i} failed: {e}")
                traceback.print_exc()
                out_records.append(
                    self.build_csv_row(
                        original_input=raw,
                        denoised_text="",
                        report_text="",
                    )
                )

            if (i - start_row + 1) % 5 == 0:
                self._save_output_csv(out_records, output_csv)

        self._save_output_csv(out_records, output_csv)
        print(f"Batch processing complete: {len(out_records)} rows saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-consultation batch processor (denoise + multi-turn pre-consultation + report)"
    )
    parser.add_argument("--input_csv", type=str, help="Input CSV; first column is raw noisy text")
    parser.add_argument("--output_csv", type=str, help="Output CSV path")
    parser.add_argument("--input_text", type=str, help="Single text input (mutually exclusive with --input_csv)")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode (terminal patient input)")
    parser.add_argument("--disable_denoising", action="store_true", help="Disable denoising")
    parser.add_argument(
        "--disable_research_scoring",
        action="store_true",
        help="Disable research scoring (not part of core service)",
    )
    parser.add_argument("--has_header", action="store_true", help="Input CSV has header row")
    parser.add_argument(
        "--encoding",
        type=str,
        default="auto",
        help="Input CSV encoding: auto (detect), utf-8-sig, gbk, gb18030, etc.",
    )
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--end_row", type=int, default=None)
    parser.add_argument("--save_report", type=str, default=None, help="Single-item mode: report output path")

    args = parser.parse_args()

    processor = BatchMedicalProcessor(config_file=args.config)
    if args.disable_denoising:
        processor.config["enable_denoising"] = False
        processor.denoiser = None
    if args.disable_research_scoring:
        processor.config["enable_research_scoring"] = False

    if args.input_csv:
        if not args.output_csv:
            print("Batch mode requires --output_csv")
            sys.exit(1)
        processor.process_csv(
            args.input_csv,
            args.output_csv,
            has_header=args.has_header,
            encoding=args.encoding,
            interactive=args.interactive,
            start_row=args.start_row,
            end_row=args.end_row,
        )
        return

    if args.input_text:
        result = processor.process_single(
            args.input_text,
            interactive=args.interactive,
            run_research_scoring=not args.disable_research_scoring,
        )
        print(result["report_text"])
        if result["dialogue_score"]:
            print("\n[Research] Overall model-reply scores:", result["dialogue_score"])
        if result["report_score"]:
            print("[Research] Report scores:", result["report_score"])
        if args.save_report:
            with open(args.save_report, "w", encoding="utf-8") as f:
                f.write(result["report_text"])
        return

    parser.print_help()


if __name__ == "__main__":
    main()
