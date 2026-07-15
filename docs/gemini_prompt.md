# Offline Gemini prompt (paper Appendix A)

For reproducibility, this is the exact prompt used to query **Gemini flash 3.1** in the offline
precomputation stage (`src/precompute_gemini.py`). It is sent **multimodally together with the meme
image, in a single API call per meme**, and covers all three subtasks at once.

- Response constrained to JSON: `response_mime_type="application/json"`.
- The four safety filters are set to `BLOCK_NONE` so the model can analyze sexist content without
  being blocked (this is analysis of harmful content for a detection task, not generation).
- `{ocr_text}` is replaced at runtime with the meme's OCR text.

The response fields are used in three ways at the fusion layer: the `description`, `sexism_analysis`
and `reasoning` strings are concatenated with the OCR and fed to the transformer encoder; the
numeric estimates become auxiliary Gemini features (7 for Task 2.2, 6 for Task 2.3); and in the
blended runs the zero-shot probabilities participate in the linear combinations (see
[`architecture.md`](architecture.md)).

```text
You are an expert annotator analyzing a meme for the EXIST 2026 shared task
at CLEF, which focuses on automatic detection of sexism in social media memes.

You will analyze this meme according to THREE complementary tasks:

TASK 1 - Binary sexism detection
A meme is SEXIST if it expresses, describes, perpetuates or criticizes sexist
behavior, stereotypes or discrimination against women.

TASK 2 - Author intention (only relevant if sexist)
- DIRECT: the author endorses or perpetuates sexism. Sexist content presented
  as message.
- JUDGEMENTAL: the author criticizes or denounces sexism. Uses irony, sarcasm,
  or contrast to expose sexist behavior. Look for contradictions between image
  and text, mocking tone, hashtags used ironically, showing absurdity of
  sexist beliefs.

TASK 3 - Categories of sexism (multi-label, can have several or none)
- IDEOLOGICAL-INEQUALITY: denies inequality, discredits feminism, claims men
  are oppressed.
- STEREOTYPING-DOMINANCE: women in submissive roles, "women's place is
  kitchen", weak/emotional women, men as dominant.
- OBJECTIFICATION: women reduced to body parts, focus on physical attributes
  only, depersonalization.
- SEXUAL-VIOLENCE: sexual harassment, assault, rape culture, unwanted sexual
  advances.
- MISOGYNY-NON-SEXUAL-VIOLENCE: hatred toward women, non-sexual aggression,
  gendered insults.

OCR text extracted from the meme: "{ocr_text}"

INSTRUCTIONS:
1. Examine image and text carefully.
2. Pay attention to context, irony, sarcasm - especially contradictions
   between visual and textual elements.
3. A meme can have multiple categories simultaneously (Task 3).
4. If the meme is NOT sexist, set task2_2.intention to "NO" and
   task2_3.categories_present to [].

Return ONLY a valid JSON object:
{
  "description": "<brief literal description of image and text>",
  "sexism_analysis": "<analysis of why or why not sexist>",
  "reasoning": "<step-by-step reasoning>",
  "task2_1": {"sexist_probability": <float 0.0-1.0>, "confidence": <float 0.0-1.0>},
  "task2_2": {
    "intention": "<NO | DIRECT | JUDGEMENTAL>",
    "intention_probabilities": {"NO": <float>, "DIRECT": <float>, "JUDGEMENTAL": <float>},
    "intention_reasoning": "<why this intention>",
    "irony_detected": <true | false>, "irony_confidence": <float 0.0-1.0>
  },
  "task2_3": {
    "categories_present": [<list of categories that apply, can be empty>],
    "category_probabilities": {
      "IDEOLOGICAL-INEQUALITY": <float>, "STEREOTYPING-DOMINANCE": <float>,
      "OBJECTIFICATION": <float>, "SEXUAL-VIOLENCE": <float>,
      "MISOGYNY-NON-SEXUAL-VIOLENCE": <float>
    },
    "category_reasoning": "<justification for each category>"
  }
}
intention_probabilities must sum to 1.0. Category probabilities are
independent (each in 0.0-1.0).
```
