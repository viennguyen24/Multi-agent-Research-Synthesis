from __future__ import annotations

PRESENTBENCH_JUDGE_SYSTEM_PROMPT = """You are an exacting slide-deck judge.

Evaluate only the requested checklist item against the provided candidate slides and, when included,
the source background. Do not infer missing evidence. If the evidence is ambiguous, answer NO.

Return your final answer in this exact format:
\\boxed{YES}
or
\\boxed{NO}

After the boxed verdict, include a brief explanation grounded in the provided evidence."""


def build_presentbench_user_prompt(checklist_item: str, source_text: str | None) -> str:
    sections = [
        "Checklist item to verify:",
        checklist_item.strip(),
    ]
    if source_text:
        sections.extend(
            [
                "",
                "Background source material:",
                source_text.strip(),
            ]
        )
    sections.extend(
        [
            "",
            "Judge only this single checklist item.",
            "Answer with a boxed YES or NO, then a short explanation.",
        ]
    )
    return "\n".join(sections)
