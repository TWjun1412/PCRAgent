# PCRAgent

Analysis Code for: **[PCRAgent: A Multi-Agent clinical pre-consultation system for structuring noisy patient reported information into clinical reports and AI-ready data]**

# 🌟 Overview

This repository contains the official implementation of **PCRAgent**, a collaborative multi-agent framework for clinical dialogue processing. The codebase is organized around two primary entry points:

| Script | Role |
| --- | --- |
| **`medical_denoising_agent.py`** | **PCRAgent denoising pipeline** — rigorously cleans raw medical dialogues via a Parallel Feedback Strategy and produces a **Traceable EditList** for clinical transparency. |
| **`batch_medical_processor.py`** | **End-to-end pre-consultation pipeline** — denoises input, runs multi-turn pre-consultation staff dialogue, generates a structured **Pre-Consultation Report (PCR)**, and optionally performs research scoring. |

Unlike generic LLMs, PCRAgent employs specialized detectors for five categories of semantic noise, resolves edit conflicts through multi-agent arbitration, and iteratively refines output via **G-Eval** quality scoring. Even under 100% noise perturbation, the framework achieves near-perfect restoration performance, outperforming generic LLMs like GPT-4o.

**End-to-end workflow:**

```
Raw noisy text
    → [medical_denoising_agent]  Detection → Editing → Arbitration → Quality loop
    → [batch_medical_processor]  Input classification → Multi-turn pre-consultation → PCR report → (optional) research scoring
```

---

# 📂 Repository Structure

## Core Scripts

| File | Description |
| --- | --- |
| `medical_denoising_agent.py` | PCRAgent denoising framework; main class: `DetectorEditorArbiter` |
| `batch_medical_processor.py` | Pre-consultation batch / single-item processor; main class: `BatchMedicalProcessor` |
| `pcr_agent_prompt.py` | LLM prompt templates for denoising (`PCRAgentPrompts`) |
| `pre_consultation_prompts.py` | LLM prompt templates for pre-consultation dialogue and report generation |
| `llm_api_utils.py` | Shared OpenAI-compatible API client, retry, and token utilities |
| `config.json` | API key, model paths, thresholds, and runtime settings |
| `medical_terms.json` | Protected medical dictionary for spell-checking |

---

## Part 1 — `medical_denoising_agent.py` (PCRAgent Denoising Pipeline)

The scripts mirror the four core functional modules described in the manuscript:

### 1. Error Detector Module (Parallel Feedback Strategy)

This module simultaneously identifies five categories of semantic noise to ensure no clinical information is misinterpreted.

| Class Name | Noise Category | Description | Key Methods/Packages |
| --- | --- | --- | --- |
| `SpellChecker` | **Spelling (SPL)** | Detects ASR-induced typos while protecting professional medical terms. | `symspellpy`, `MedicalTermsManager` |
| `RepetitionDetector` | **Repetition (RPT)** | Identifies redundant oral fragments and n-gram stutters. | `collections.Counter`, `re` |
| `GECTagger` | **Grammar (GRM)** | Corrects syntactic fragmentation using seq2seq models. | `transformers` (GEC Model) |
| `CombinedMedicalDetector` | **Ambiguity (AMB) & Interference (NOS)** | LLM-powered detection of ambiguous clinical terms and non-medical noise. | OpenAI-compatible API, `PCRAgentPrompts` |

**Main orchestrator:** `DetectorEditorArbiter` — wires all detectors, editor, arbiter, and quality evaluator.

### 2. Semantic Editing Module (Traceable EditList)

Translates detection signals into precise modifications while maintaining a verifiable audit trail.

| Class Name | Description | Key Features |
| --- | --- | --- |
| `EditManager` | Organizes edits into the **Traceable EditList**, calculating confidence and cost for each change. | Edit tracking, cost scoring, span-level ops (`REPLACE` / `DELETE` / `INSERT`) |
| `EditorPipeline` | Executes targeted semantic restoration, focusing on clinical Word Sense Disambiguation (WSD). | Deterministic vs. candidate edit routing |

Each edit in the Traceable EditList follows this schema:

```python
{
  "start_char": int,
  "end_char": int,
  "op": "REPLACE" | "DELETE" | "INSERT",
  "cand_texts": ["candidate1", "candidate2", ...],
  "score": float,
  "tag": "RPT|SPL|GRM|AMB|NOS",
  "edit_type": "deterministic|candidate"
}
```

### 3. Output Control Module (Iterative Quality Loop)

A multi-agent arbitration system that resolves edit conflicts and ensures output standards.

| Class Name | Description | Key Methods/Packages |
| --- | --- | --- |
| `ArbiterCore` / `ArbiterPipeline` | Resolves conflicts between overlapping edit candidates using a priority-based selection strategy. | Conflict resolution, final text synthesis |
| `DenoisingQualityGEval` | Performs multi-dimensional quality assessment (Accuracy, Integrity, Smoothness) to trigger iterative refinement. | **G-Eval Framework**, up to 3 reprocess rounds |

### 4. Evaluation Module

| Class Name | Description | Key Metrics |
| --- | --- | --- |
| `EvaluationMetrics` | Comprehensive calculation of term retention rates and semantic similarity. | Term retention, Cohen's Kappa |
| `MedicalTermsManager` | Loads and manages the protected medical dictionary. | Dictionary lookup, term protection |

---

## Part 2 — `batch_medical_processor.py` (Pre-Consultation Pipeline)

Extends denoising with a multi-turn pre-consultation workflow and structured report generation.

### Pipeline Stages

| Stage | Method | Description |
| --- | --- | --- |
| **1. Denoising** | `denoise_text()` | Invokes `DetectorEditorArbiter.denoise()` on raw input |
| **2. Input classification** | `classify_input()` | Distinguishes `complete_dialogue` vs. `patient_utterance` |
| **3. History extraction** | `extract_modules_from_text()` | Extracts HPI, past history, allergy, family history, etc. |
| **4. Multi-turn dialogue** | `run_pre_consultation_dialogue()` | Staff-guided collection with inner-world reconstruction |
| **5. Report generation** | `build_report()` | Produces a structured **Pre-Consultation Report (PCR)** |
| **6. Research scoring** *(optional)* | `score_dialogue_replies()`, `score_report()` | Accuracy, Completeness, Security, Clarity |

### Key Classes

| Class Name | Description | Key Features |
| --- | --- | --- |
| `BatchMedicalProcessor` | Top-level orchestrator for single-item and CSV batch processing | Config loading, denoiser init, LLM chat wrapper |
| `PreConsultationSession` | Stateful multi-turn dialogue tracker | Phase management (`COLLECTING` → `FINISHED`), transcript logging |
| `HistoryModules` | Structured history fields | Chief complaint, HPI, past/allergy/family/personal history |
| `PreConsultationReport` | Final report dataclass | Module sections + conversation summary |
| `SessionPhase` | Dialogue state machine | Collecting, completion ack, end confirmation |

### Batch CSV Output Columns

| Column | Description |
| --- | --- |
| `Original input` | Raw noisy input text |
| `Denoised text` | Output after PCRAgent denoising |
| `Pre-consultation report` | Structured PCR report |
| `Model reply_*` | Research scores for staff dialogue (Accuracy, Completeness, Security, Clarity) |
| `Report_*` | Research scores for the generated report |

---

# 🛠️ Prerequisites & Installation

Requires **Python >= 3.8** and an OpenAI-compatible API key.

```bash
# Clone / enter project directory
cd DEA_agent

# Install dependencies
pip install -r requirements.txt

# Register local modules (recommended for IDE import resolution)
pip install -e .

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

**Configuration:** Copy and edit `config.json`:

```json
{
  "api_key": "your-api-key",
  "base_url": "https://api.chatanywhere.tech/v1",
  "chat_model": "deepseek-v3",
  "medical_dictionary_path": "medical_terms.json",
  "api_timeout": 300
}
```

**Note:** LLM-powered modules (`CombinedMedicalDetector`, `EditorPipeline`, `ArbiterPipeline`, `DenoisingQualityGEval`, and the entire pre-consultation pipeline) require a valid API key. Local models (GEC, SymSpell) can run without an API key for deterministic detectors only.

**Optional local model paths** (configure in `config.json` → `model_paths`):

- `gec_model` — Grammar Error Correction seq2seq model
- `glossbert_model` — GlossBERT for WSD
- `simcse_model` — Sentence embedding model

---

# 🚀 Quick Start

## A. Denoising Only (`medical_denoising_agent.py`)

```python
from medical_denoising_agent import DetectorEditorArbiter

# Initialize with medical dictionary and API key
agent = DetectorEditorArbiter(
    medical_dictionary_path="medical_terms.json",
    api_key="your-openai-api-key",
    base_url="https://api.chatanywhere.tech/v1",
)

# Process a noisy clinical dialogue
raw_dialogue = (
    "Pt: I... I feel a bit, uh, cough, cough... is it COVID? "
    "(background noise: barking)"
)

# Detection → Editing → Arbitration → Quality loop
result = agent.denoise(raw_dialogue, verbose=True)

print(f"Restored Text: {result['final_text']}")
print(f"Traceable EditList: {result['resolved_edits']}")
print(f"Quality Scores: {result['quality_scores']}")
```

## B. Full Pre-Consultation Pipeline (`batch_medical_processor.py`)

```python
from batch_medical_processor import BatchMedicalProcessor

processor = BatchMedicalProcessor(config_file="config.json")

result = processor.process_single(
    "Patient: I've had a headache for three days with mild fever...",
    interactive=False,          # Set True for terminal patient input
)

print(result["denoised_text"])
print(result["report_text"])
print(result["dialogue_score"])   # Optional research metrics
print(result["report_score"])
```

## C. Command-Line Interface

```bash
# Single text → pre-consultation report
python batch_medical_processor.py \
  --config config.json \
  --input_text "Patient: headache and fever for 3 days..." \
  --save_report report.txt

# Interactive mode (type patient replies in terminal)
python batch_medical_processor.py \
  --config config.json \
  --input_text "Patient: chest pain since yesterday" \
  --interactive

# CSV batch processing (first column = raw noisy input)
python batch_medical_processor.py \
  --config config.json \
  --input_csv input.csv \
  --output_csv output.csv \
  --has_header

# Skip denoising or research scoring
python batch_medical_processor.py \
  --input_csv input.csv \
  --output_csv output.csv \
  --disable_denoising \
  --disable_research_scoring
```

### CLI Reference

| Flag | Description |
| --- | --- |
| `--input_text` | Single raw text input |
| `--input_csv` / `--output_csv` | Batch CSV input / output paths |
| `--config` | Path to `config.json` (default: project root) |
| `--interactive` | Terminal-based patient replies |
| `--disable_denoising` | Skip PCRAgent denoising stage |
| `--disable_research_scoring` | Skip research evaluation columns |
| `--has_header` | Input CSV includes a header row |
| `--encoding` | CSV encoding (`auto`, `utf-8-sig`, `gbk`, etc.) |
| `--start_row` / `--end_row` | Process a subset of CSV rows |
| `--save_report` | Save single-item report to file |

---

# 📊 Performance Metrics

As validated in the study across 220,000+ dialogues:

* **Restoration Accuracy**: 4.995/5.000 (at 100% noise)
* **Medical Term Retention**: Significantly higher than standard GPT-4 models
* **Clinical Safety**: Zero-hallucination restoration via the Traceable EditList
* **Denoising quality dimensions**: Accuracy, Integrity, Smoothness (G-Eval, threshold-triggered reprocessing)

---

# 📜 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{pcragent2026,
  title={PCRAgent: A Multi-Agent clinical pre-consultation system for structuring noisy patient reported information into clinical reports and AI-ready data},
  author={...},
  year={2026}
}
```

---

*This repository is for research purposes. Ensure compliance with local regulations regarding medical data (e.g., HIPAA) when using LLMs.*
