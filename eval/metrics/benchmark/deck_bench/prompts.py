"""Prompt text imported from DeckBench.

Source:
https://github.com/morgan-heisler/DeckBench/blob/main/metrics/prompts.py
"""

coherence_system_prompt = """You are an unbiased presentation analysis judge responsible for evaluating the coherence of the 
presentation. Please carefully review the provided summary of the presentation, assessing its logical flow
and contextual information, each score level requires that all evaluation criteria meet the standards of that level."""

content_system_prompt = """You are an unbiased presentation analysis judge responsible for evaluating the quality of slide content.
Please carefully review the provided slide image, assessing its content, and provide your judgement in a 
JSON object containing the reason and score. Each score level requires that all evaluation criteria meet the 
standards of that level."""

coherence_prompt = """Scoring Criteria (Five-Point Scale)
1 Point (Poor):
Terminology are inconsistent, or the logical structure is unclear, making it difficult for the audience to
understand.
2 Points (Fair):
Terminology are consistent and the logical structure is generally reasonable, with minor issues in
transitions.
3 Points (Average):
The logical structure is sound with fluent transitions; however, it lacks basic background information.
4 Points (Good):
The logical flow is reasonable and include basic background information (e.g., speaker or
acknowledgments/conclusion).
5 Points (Excellent):
The narrative structure is engaging and meticulously organized with detailed and comprehensive
background information included.
Example Output:
{{
"reason": "xx",
"score": int
}}
Input:
{slides_gen}
Let's think step by step and provide your judgment, focusing exclusively on the dimensions outlined above
and strictly follow the criteria.
    """

content_prompt = """Scoring Criteria (Five-Point Scale):
1 Point (Poor):
The text on the slides contains significant grammatical errors or is poorly structured, making it difficult to
understand.
2 Points (Below Average):
The slides lack a clear focus, the text is awkwardly phrased, and the overall organization is weak, making it
hard to engage the audience.
3 Points (Average):
The slide content is clear and complete but lacks visual aids, resulting in insufficient overall appeal.
4 Points (Good):
The slide content is clear and well-developed, but the images have weak relevance to the theme, limiting
the effectiveness of the presentation.
5 Points (Excellent):
The slides are well-developed with a clear focus, and the images and text effectively complement each
other to convey the information successfully.
Example Output:
{{
"reason": "xx",
"score": int
}}
Input: {slides_gen}
Let's think step by step and provide your judgment."""

