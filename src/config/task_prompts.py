CHARTMUSEUM_PROMPTS = {
    "QA": """"Please answer the question using the chart image.

Question: [QUESTION]

Please first provide a brief rationale and then provide the answer. Use the following format:

<rationale> 
... your rationale here ...
</rationale> 
<answer> 
... your final answer (entity(s) or number) ...
</answer>""",
   
   "COMPARE_ANSWER": """You are provided with a question and two answers. Please determine if these answers are equivalent. Follow these guidelines:

1. Numerical Comparison:
   - For decimal numbers, consider them as equivalent if their relative difference is sufficiently small. 
   For example, the following pairs are equivalent:
    - 32.35 and 32.34
    - 90.05 and 90.00
    - 83.3% and 83.2%
    - 31 and 31%
   The following pairs are not equivalent:
   - 32.35 and 35.25
   - 90.05 and 91.05
   - 83.3% and 45.2%

   - If the question asks for a value read approximately from a chart, accept reasonable approximations, including words such as "around", "approximately", "early", "mid", "late", or short ranges, when they refer to the same visible location or tick interval.
   - For years read from an axis or threshold crossing in a chart, accept nearby approximations that are visually indistinguishable at the chart resolution. For example, "around 1970" and "1970" are equivalent, and "early 1970s" may be equivalent to a crossing near 1970--1975.
   - For exact calendar dates or explicitly labeled years where the question requires an exact date, require an exact match.

2. Unit Handling:
   - If only one answer includes units (e.g. '$', '%', '-', etc.), ignore the units and compare only the numerical values
   For example, the following pairs are equivalent:
   - 305 million and 305 million square meters
   - 0.75 and 0.75%
   - 0.6 and 60%
   - $80 and 80
   The following pairs are not equivalent:
   - 305 million and 200 million square meters
   - 0.75 and 0.90%

3. Text Comparison:
   - Ignore differences in capitalization
   - Treat mathematical expressions in different but equivalent forms as the same (e.g., "2+3" = "5")
   - Treat minor spelling, punctuation, spacing, hyphenation, underscore, and LaTeX-formatting differences as equivalent when they refer to the same entity, method, curve, category, or symbol.
   - If the gold answer lists multiple acceptable alternatives using "or", "/", commas, or similar separators, the prediction is correct if it gives one of those alternatives and does not add a contradictory answer.
   - If the question asks for multiple items, the prediction must include the required set without adding contradictory extra items. Ignore ordering unless the question asks for an order.
   - If the prediction includes a rationale plus a final answer, judge only the final answer unless the final answer is missing or ambiguous.

Question: [QUESTION]
Answer 1: [ANSWER1]
Answer 2: [ANSWER2]

Please respond with:
- "Yes" if the answers are equivalent
- "No" if the answers are different"""
}
