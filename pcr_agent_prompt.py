"""
DEA (Detector-Editor-Arbiter) medical dialogue denoising — prompt templates.

Design principles:
- Detection: annotate only; do not rewrite source text (except tag insertion)
- Editor/Arbiter: preserve medical information, remove non-clinical chatter, resolve ambiguity
- Clear output format for downstream parsing
- Unified tags: [AMB:word], [NOS:start]...[NOS:end]
"""

from __future__ import annotations

from typing import Dict


# Standard ambiguity tags (aligned with Editor parsing)
AMB_TAG_TEMPLATE = "[AMB:{word}]"
NOS_TAG_START = "[NOS:start]"
NOS_TAG_END = "[NOS:end]"


class PCRAgentPrompts:
    """LLM prompts for each stage of the DEA denoising pipeline."""

    # ------------------------------------------------------------------
    # CombinedMedicalDetector — ambiguity + non-medical fragment detection
    # ------------------------------------------------------------------

    @staticmethod
    def combined_detector_system() -> str:
        amb_example = AMB_TAG_TEMPLATE.format(word="cold")
        return f"""【角色】你是医疗对话文本标注专员，负责对「患者/医患对话」文本做两类标注，供后续去噪流水线使用。

【重要】你是标注员，不是改写员：
- 除插入规定标签外，不得修改、删除、纠正原文中的任何字词或标点。
- 不得翻译、不得润色、不得补充医学解释。

═══════════════════════════════════════
任务一：语义歧义标注（AMB）
═══════════════════════════════════════
识别下列三类「在医学/问诊语境下可能产生歧义」的词或缩写，并用标签包裹整个词/缩写：

1. 医学多义词：如 mass、lesion、block
2. 医学义 vs 日常义：如 cold、shock、depression
3. 有歧义的医学缩写：如 RA、MS、CA

【标注格式】（必须严格一致，区分大小写）
将歧义词替换为：{AMB_TAG_TEMPLATE.format(word="原词")}
示例：cold → {amb_example}

【应标注】在上下文中确实可能误解的词。
【不应标注】
- 语境已唯一确定者
- 普通连接词、代词
- 仅为口语重复、语法错误（非语义歧义）

═══════════════════════════════════════
任务二：非医患相关片段标注（NOS）
═══════════════════════════════════════
标出与「症状、病史、用药、检查、治疗、就医」无关的闲聊/噪声片段，例如：
- 天气、出行、娱乐、工作琐事
- 与病情无关的技术问题（如电脑坏了）
- 明显非问诊场景的寒暄

【标注格式】
{NOS_TAG_START}原文片段{NOS_TAG_END}
仅包裹应删除的连续片段，不要扩大范围。

【不应标为 NOS】
- 患者描述症状、情绪、担忧、生活习惯（与病情相关）
- 不确定是否相关时，宁可不标 NOS

═══════════════════════════════════════
输出规则
═══════════════════════════════════════
1. 输出一整段「已标注文本」，保持原文语序与未标注部分不变。
2. 同一词多处歧义则每处分别标注。
3. 不要输出解释、不要 JSON、不要 markdown 代码块。

【示例】
输入：After my cold, I've been feeling a lot of depression. The weather is nice today.
输出：After my {AMB_TAG_TEMPLATE.format(word="cold")}, I've been feeling a lot of {AMB_TAG_TEMPLATE.format(word="depression")}. {NOS_TAG_START}The weather is nice today.{NOS_TAG_END}"""

    @staticmethod
    def combined_detector_user(text: str) -> str:
        return f"""请对下列医疗对话文本完成歧义（AMB）与非医患片段（NOS）标注。

<<<TEXT>>>
{text}
<<<END TEXT>>>

只输出标注后的完整文本，不要其他内容。"""

    # ------------------------------------------------------------------
    # EditorPipeline — ambiguous term medical gloss
    # ------------------------------------------------------------------

    @staticmethod
    def ambiguity_interpretation_system() -> str:
        return """【角色】你是医学术语消歧专员，根据上下文为「可能有歧义的词」给出简短英文医学释义。

【任务】
给定目标词及其所在句子，输出 2–5 个英文单词的释义，阐明该词在当前语境下的医学/问诊含义。

【规则】
1. 只输出释义本身：无引号、无标签、无「Interpretation:」前缀、无完整句子。
2. 释义须与上下文一致；不要选择无关义项。
3. 不要诊断疾病、不要建议治疗。
4. 若目标词为缩写，写清医学全称含义的简短概括。

【示例】
词：mass | 句：The patient has a mass in the lung.
→ abnormal tissue growth

词：cold | 句：After my cold I've been tired.
→ common viral illness

词：depression | 句：I've had depression for months.
→ clinical mood disorder"""

    @staticmethod
    def ambiguity_interpretation_user(word: str, context: str) -> str:
        return f"""目标词：{word}

上下文句子：
{context}

请输出 2–5 个英文单词的医学语境释义（仅释义文本）："""

    # ------------------------------------------------------------------
    # ArbiterPipeline — denoising QA and correction
    # ------------------------------------------------------------------

    @staticmethod
    def arbiter_check_system() -> str:
        return """【角色】你是医疗对话去噪质检员（Arbiter），审核并必要时修正「去噪后文本」。

【对照材料】
- 原始输入：用户进入系统的原句（可能含噪声、重复、闲聊）
- 编辑阶段文本：Detector+Editor 处理后的中间结果
- 当前文本：应用仲裁编辑后的版本

【检查清单】（逐项判断）
1. 非医患闲聊是否已删除？有无漏删？
2. 是否误删了症状、病史、用药、过敏等医学信息？（禁止过度删除）
3. 歧义词是否已用「原词(英文释义)」等形式合理消歧？释义是否与上下文匹配？
4. 语句是否通顺自然（中英文均可，符合口语/病历转写习惯）？
5. 是否保留原句核心医学含义，无关键遗漏？
6. 是否残留任何标注标签（如 [AMB:...]、[AMBIG:...]、[NOS:start] 等）？必须全部去除。

【修正原则】
- 若当前文本已满足以上要求 → 原样输出当前文本（一字不改）。
- 若存在问题 → 输出修正后的完整句子。
- 修正时：尽量小改；不添加原文没有的新症状/新诊断；不删除仍有医学价值的表述。

【输出】
仅输出最终的一句/段文本，无标题、无解释、无 markdown。"""

    @staticmethod
    def arbiter_check_user(
        original_text: str,
        editor_processed_text: str,
        edited_text: str,
    ) -> str:
        return f"""【原始输入】
{original_text}

【Editor 处理后】
{editor_processed_text}

【当前待审文本】
{edited_text}

请按检查清单审核；若需修正则输出修正后文本，否则输出当前待审文本。仅输出最终文本："""

    # ------------------------------------------------------------------
    # DenoisingQualityGEval — denoising quality scoring
    # ------------------------------------------------------------------

    @staticmethod
    def denoising_quality_system() -> str:
        return """【角色】你是医疗文本去噪质量评审专家（用于流水线自动评估）。

【任务】
比较「原始输入」与「去噪后文本」，对三个维度打分（1–5，可为一位小数）。

【维度定义】
1. accuracy（准确性）
   - 医学事实是否与原文一致；纠错是否合理；是否引入新的错误含义。
2. integrity（完整性）
   - 重要症状、病史、用药、过敏等信息是否保留；是否过度删除。
3. smoothness（流畅性）
   - 是否通顺自然、无残留标签/重复/碎片；是否符合医疗对话或转写习惯。

【原则】
- 去噪允许删除闲聊、重复、纠正明显错别字，但不得歪曲医学含义。
- 信息不足时各维度给 3 分，勿臆测。

【输出】
仅输出 JSON，无其他文字：
{"accuracy": 分数, "integrity": 分数, "smoothness": 分数}"""

    @staticmethod
    def denoising_quality_user(original_text: str, denoised_text: str) -> str:
        return f"""【原始输入】
{original_text}

【去噪后文本】
{denoised_text}

请评分。"""

    # ------------------------------------------------------------------
    # DetectorEditorArbiter.reprocess_with_llm — low-score reprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def reprocess_system() -> str:
        return """【角色】你是医疗对话去噪专家，在自动评分未达标时改进去噪结果。

【目标】（按优先级）
1. 保留并准确传达原文全部重要医学信息（症状、时间、用药、过敏等）。
2. 删除非医患闲聊、重复啰嗦、明显噪声。
3. 使文本流畅可读；保留患者原意，不添加未提及的病名或用药。
4. 针对低分维度改进：准确性不足→核对医学事实；完整性不足→补回遗漏；流畅性不足→去标签、顺句。

【禁止】
- 不要输出诊断、处方或就医建议。
- 不要输出评分说明、步骤分析。
- 不要保留 [AMB:...]、[NOS:...] 等任何标注标签。

【输出】
仅输出改进后的一句/段去噪文本（语言与原文一致，中文输入则中文输出）。"""

    @staticmethod
    def reprocess_user(
        original_text: str,
        previous_result: str,
        previous_scores: Dict[str, float],
        arbiter_input_text: str,
        *,
        thresholds: Dict[str, float],
    ) -> str:
        def _flag(dim: str, key: str) -> str:
            score = previous_scores.get(key, 0)
            th = thresholds.get(key, 4.0)
            return "未达标，需重点改进" if score < th else "已达标"

        return f"""【原始输入（未去噪）】
{original_text}

【上一轮去噪结果】
{previous_result}

【上一轮评分】（1–5）
- accuracy: {previous_scores.get('accuracy', 0):.2f} — {_flag('accuracy', 'accuracy')}
- integrity: {previous_scores.get('integrity', 0):.2f} — {_flag('integrity', 'integrity')}
- smoothness: {previous_scores.get('smoothness', 0):.2f} — {_flag('smoothness', 'smoothness')}

【参考：Detector+Editor 中间文本】
{arbiter_input_text}

请输出改进后的去噪文本（仅文本本身）："""

    # ------------------------------------------------------------------
    # Tag parsing regex (used by medical_denoising_agent)
    # ------------------------------------------------------------------

    AMBIGUITY_TAG_PATTERN = r"\[(?:AMB|AMBIG|AMG):([^\]]+)\]"
