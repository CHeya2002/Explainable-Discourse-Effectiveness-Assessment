"""Narrative-based explanation and storytelling generation.

Converted from the original research notebook while preserving the original code-cell execution order.
"""


# %% [Original notebook cell 1]
from pathlib import Path
import json
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 300)

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Final storytelling input.
# This repository exposes only the final storytelling stage; intermediate
# planning scripts are intentionally not part of the public pipeline.
#
# Each record in this JSON file must already contain the final engine-controlled
# fields consumed by the unchanged story-generation code, including story_plan,
# probabilities, XAI summary, labels, and discourse information.
STORYTELLING_INPUT_PATH = PROJECT_ROOT / "data" / "storytelling_inputs.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "storytelling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_JSON_PATH = OUTPUT_DIR / "final_storytelling_results_v2.json"
FULL_STORIES_CSV_PATH = OUTPUT_DIR / "full_stories_v2.csv"
VISIBLE_PLANS_CSV_PATH = OUTPUT_DIR / "visible_story_plans_v2.csv"
VALIDATION_CSV_PATH = OUTPUT_DIR / "story_validation_v2.csv"
DASHBOARD_CSV_PATH = OUTPUT_DIR / "storytelling_dashboard_ready_v2.csv"
QUALITY_REPORT_PATH = OUTPUT_DIR / "step3_quality_report_v2.json"
EXCEL_PATH = OUTPUT_DIR / "step3_storytelling_results_v2.xlsx"

# ============================================================
# OLLAMA
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

REQUEST_TIMEOUT = 600
MAX_RETRIES = 1
RETRY_WAIT_SECONDS = 2

USE_OLLAMA = True
ALLOW_FALLBACK = True
RETRY_ON_VALIDATION_FAILURE = True

TEMPERATURE = 0.20
TOP_P = 0.90
REPEAT_PENALTY = 1.10
NUM_PREDICT = 1200

CLASS_LABELS = ["Ineffective", "Adequate", "Effective"]

if not STORYTELLING_INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Missing final storytelling input file:\n{STORYTELLING_INPUT_PATH}\n"
        "Provide data/storytelling_inputs.json containing the final "
        "engine-controlled inputs used by the narrative generator."
    )

with open(STORYTELLING_INPUT_PATH, "r", encoding="utf-8") as f:
    realization_inputs = json.load(f)

if not isinstance(realization_inputs, list) or not realization_inputs:
    raise ValueError("storytelling_inputs.json must contain a non-empty JSON list.")

# The generation/export code expects a lookup of the already-selected plans.
story_plan_map = {
    int(item["example_id"]): item["story_plan"]
    for item in realization_inputs
}

print("Realization inputs :", len(realization_inputs))
print("Model              :", OLLAMA_MODEL)
print("Output folder      :", OUTPUT_DIR)

# %% [Original notebook cell 2]
# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def stringify_list(values, separator=" | "):
    if values is None:
        return ""
    if isinstance(values, list):
        return separator.join(str(x) for x in values)
    return str(values)

def extract_json_object(text):
    if not text:
        raise ValueError("Empty response.")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No complete JSON object found.")

    parsed = json.loads(cleaned[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Parsed result is not a JSON object.")

    return parsed

def check_ollama_connection():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )
        response.raise_for_status()
        payload = response.json()

        models = [
            item.get("name")
            for item in payload.get("models", [])
        ]

        return {
            "connected": True,
            "available_models": models,
            "requested_model_available": OLLAMA_MODEL in models
        }
    except Exception as error:
        return {
            "connected": False,
            "available_models": [],
            "requested_model_available": False,
            "error": str(error)
        }

ollama_status = check_ollama_connection()
print(json.dumps(ollama_status, indent=2, ensure_ascii=False))

# %% [Original notebook cell 3]
# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
\human You are the narrative realization component of a rule-guided
educational storytelling system.

The engine has already selected the complete story structure.
You must not select, remove, replace, or reorder story elements.

Your role is to REALIZE the supplied evidence as a concise educational
narrative, not to create new evidence, explanations, examples, or assumptions.

Write one coherent educational narrative that follows the supplied
narrative order naturally.

Mandatory requirements:

1. Begin with the supplied hook.

2. Introduce the selected hero and its supplied goal without adding
   motivations, intentions, or characteristics that are not provided.

3. Present the supplied assessment context.

4. Introduce the supplied inciting event.

5. Let the selected sidekick support the explanation using ONLY the
   evidence explicitly associated with it in the input.
   Do not invent evidence merely to make the sidekick narratively meaningful.

6. Present the strongest supplied evidence supporting the main
   interpretation. Prioritize the most informative evidence rather than
   listing every available value.

7. Present supplied limitations and competing evidence before the climax.
   When evidence supports a competing class, describe exactly what the
   supplied XAI results show. Do not create semantic explanations that
   are not supported by the input.

8. Discuss Ineffective, Adequate, and Effective explicitly, but focus
   primarily on the classes that meaningfully compete in this example.

9. Preserve all supplied prediction probabilities accurately.
   Never reinterpret LIME weights or IG attributions as probabilities.

10. Explain why the predicted class remains strongest using ONLY the
    supplied probabilities, XAI evidence, rubric information, and
    assessment facts.

11. Explicitly mention whether the predicted class agrees or disagrees
    with the reference label.

12. End with the supplied resolution and improvement target.
    Any improvement recommendation must be directly grounded in the
    supplied rubric criteria. Do not generate generic or external advice.

13. Treat LIME weights as local feature-importance values for the
    corresponding class, not as probabilities.

14. When LIME information is available for multiple classes, preserve
    meaningful cross-class comparisons. If an important token supports
    more than one class or opposes another class, communicate this when
    it helps explain the prediction.

15. Treat IG attributions as contributions to the specified target output
    relative to the supplied baseline. Positive and negative IG values
    must be interpreted only with respect to that target.

16. Treat IG delta as a completeness diagnostic, not as model confidence.

17. When LIME and IG identify the same important feature, their agreement
    may be used to strengthen the explanation. When they differ, describe
    the difference without forcing agreement.

18. Never claim that one isolated token proves, determines, or directly
    corresponds to a rubric level.

19. Do not infer missing essay context, student intentions, causal
    relationships, educational facts, or real-world explanations.

20. Do not introduce any fact, interpretation, example, or recommendation
    that cannot be traced to the supplied model outputs, XAI evidence,
    discourse text, assessment facts, or rubric.

21. If the supplied evidence is insufficient to support a narrative
    interpretation, state the uncertainty briefly rather than filling
    the gap with a plausible explanation.
22. Do not expose, name, announce, or explicitly describe the underlying
    narrative structure. The narrative framework is an internal generation
    guide only and must remain invisible to the reader. Never use structural
    labels or meta-narrative phrases such as Hook, Hero, Goal, Sidekick,
    Context, Inciting Event, Conflict, Evidence, Limitation, Climax,
    Resolution, or Improvement Target. Do not write phrases such as
    "the hero of this story," "the sidekick," "the inciting event,"
    "the central conflict," "at the climax," or "the resolution."
    Integrate all supplied elements naturally into the educational narrative
    without revealing that they correspond to predefined story components.
Return only the final reader-facing educational narrative.
The predefined narrative structure must remain completely implicit.
Do not mention the storytelling framework, narrative stages, structural
roles, generation process, system instructions, or input field names.
Use natural transitions so that the supplied structural elements appear
as part of a coherent explanation rather than as visible story stages.
Return one continuous paragraph-based story only.

23. Write approximately 200 to 350 words.

24. Preserve the essential supplied narrative structure, but keep each
    component concise and avoid repeating the same evidence or conclusion.

25. Prioritize:
    - the model prediction and competing class(es),
    - the most informative XAI evidence,
    - the central conflict,
    - the decisive evidence at the climax,
    - and the rubric-grounded resolution.

26. Do not repeat the same token values in multiple narrative stages
    unless necessary for understanding the climax.

27. Return only the narrative text.

28. Do not return JSON.

29. Do not use markdown headings or bullet points.

""".strip()

print(SYSTEM_PROMPT)

# %% [Original notebook cell 4]
# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(realization_input):
    return (
        SYSTEM_PROMPT
        + "\n\nCOMPACT ENGINE-GENERATED INPUT:\n"
        + json.dumps(
            realization_input,
            ensure_ascii=False,
            indent=2
        )
    )

preview_prompt = build_prompt(realization_inputs[0])
print("Prompt characters:", len(preview_prompt))
print(preview_prompt[:5000])

# %% [Original notebook cell 5]
# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(prompt):
    """
    Call Ollama and return the generated narrative as plain text.

    This version intentionally does not request JSON. The previous JSON
    approach could be truncated before the closing brace, even when the
    narrative itself was otherwise usable.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repeat_penalty": REPEAT_PENALTY,
            "num_predict": NUM_PREDICT
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            "Cannot connect to Ollama on localhost:11434."
        ) from error
    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            f"Ollama exceeded {REQUEST_TIMEOUT} seconds."
        ) from error

    if not response.ok:
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}:\n{response.text}"
        )

    outer = response.json()
    raw_text = str(outer.get("response", "")).strip()

    if not raw_text:
        raise RuntimeError("Ollama returned an empty response.")

    # Remove accidental code fences or labels if the model adds them.
    cleaned_text = re.sub(
        r"^```(?:text|markdown)?\s*",
        "",
        raw_text,
        flags=re.IGNORECASE
    )
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()

    # Remove accidental prefixes such as "full_story:"
    cleaned_text = re.sub(
        r"^\s*(full_story|story|narrative)\s*:\s*",
        "",
        cleaned_text,
        flags=re.IGNORECASE
    ).strip()

    if not cleaned_text:
        raise RuntimeError(
            "Ollama returned a response, but no narrative text remained after cleaning."
        )

    return {
        "full_story": cleaned_text,
        "raw_response": raw_text,
        "metadata": {
            "model": outer.get("model", OLLAMA_MODEL),
            "done": outer.get("done"),
            "done_reason": outer.get("done_reason"),
            "total_duration": outer.get("total_duration"),
            "prompt_eval_count": outer.get("prompt_eval_count"),
            "eval_count": outer.get("eval_count")
        }
    }

# %% [Original notebook cell 6]
# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def deterministic_fallback(realization_input):
    plan = realization_input["story_plan"]
    probabilities = realization_input["probabilities"]

    predicted = realization_input["predicted_label"]
    reference = realization_input["reference_label"]

    rubric = realization_input["rubric_comparison"]
    xai = realization_input["xai_summary"]

    alternative = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )[1][0]

    predicted_tokens = [
        item["token"]
        for item in xai["lime_by_class"][predicted]
    ]

    ig_tokens = [
        item["token"]
        for item in xai["ig_tokens"]
    ]

    agreement_sentence = (
        f"This agrees with the reference assessment of {reference}."
        if predicted == reference
        else (
            f"This differs from the reference assessment of {reference}, "
            "so the disagreement remains visible in the interpretation."
        )
    )

    story = (
        f"{plan['hook']['content']} "
        f"At the centre of the assessment is {plan['hero']['entity']}, "
        f"whose goal is to {plan['hero']['goal']}. "
        f"The segment is evaluated across Ineffective, Adequate, and Effective, "
        f"with probabilities of {probabilities['Ineffective']:.4f}, "
        f"{probabilities['Adequate']:.4f}, and "
        f"{probabilities['Effective']:.4f}. "
        f"{plan['inciting_event']['content']} "
        f"The supporting role is played by {plan['sidekick']['entity']}, "
        f"which helps clarify the evidence without replacing the discourse itself. "
        f"For the {predicted} interpretation, LIME highlighted "
        f"{', '.join(predicted_tokens) if predicted_tokens else 'no clear tokens'}, "
        f"while Integrated Gradients highlighted "
        f"{', '.join(ig_tokens) if ig_tokens else 'no clear tokens'}. "
        f"These features indicate where the model focused, but they do not by "
        f"themselves prove rubric satisfaction. "
        f"{plan['first_plot_point']['content']} "
        f"However, {alternative} remains the closest competing interpretation. "
        f"The rubric criteria for {predicted} include "
        f"{'; '.join(rubric[predicted]['criteria'])}, while the criteria for "
        f"{alternative} include {'; '.join(rubric[alternative]['criteria'])}. "
        f"The remaining class is also considered so that all three perspectives "
        f"remain visible. "
        f"{plan['climax']['content']} {agreement_sentence} "
        f"The final model assessment is therefore {predicted}. "
        f"To move toward {plan['resolution']['improvement_target']}, the response "
        f"should be reviewed against the supplied target rubric criteria."
    )

    return {
        "full_story": re.sub(r"\s+", " ", story).strip()
    }

# %% [Original notebook cell 7]
# ============================================================
# VALIDATION
# ============================================================

def validate_story(output, realization_input):
    issues = []
    warnings = []

    story = str(output.get("full_story", "")).strip()
    lower_story = story.lower()

    if not story:
        issues.append("Missing full_story.")

    word_count = len(story.split())

    if word_count < 180:
        warnings.append("Story is shorter than preferred.")
    if word_count > 800:
        warnings.append("Story is longer than preferred.")

    for label in CLASS_LABELS:
        if label.lower() not in lower_story:
            issues.append(f"Class not discussed: {label}")

    predicted = realization_input["predicted_label"]
    reference = realization_input["reference_label"]

    if predicted.lower() not in lower_story:
        issues.append("Predicted label is missing.")

    if reference.lower() not in lower_story:
        warnings.append("Reference label is not explicit.")

    hero_terms = [
        x for x in re.findall(
            r"[a-z]+",
            realization_input["story_plan"]["hero"]["entity"].lower()
        )
        if len(x) > 2
    ]

    if hero_terms and not any(x in lower_story for x in hero_terms):
        issues.append("Selected hero is not realized.")

    probabilities = realization_input["probabilities"]

    for label, value in probabilities.items():
        accepted = {
            f"{value:.4f}",
            f"{value:.3f}",
            f"{value:.2f}",
            f"{value * 100:.1f}%",
            f"{value * 100:.0f}%"
        }

        if not any(form in story for form in accepted):
            warnings.append(
                f"Probability for {label} is not explicit."
            )

    return {
        "validation_passed": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "character_count": len(story)
    }

# %% [Original notebook cell 8]
# ============================================================
# GENERATE ONE STORY
# ============================================================

def generate_one_story(realization_input):
    prompt = build_prompt(realization_input)

    output = None
    raw_response = ""
    metadata = {}
    generation_error = None

    if not USE_OLLAMA:
        output = deterministic_fallback(realization_input)
        source = "deterministic_fallback"
        used_fallback = True

    else:
        for attempt in range(MAX_RETRIES + 1):
            try:
                print(
                    f"Calling Ollama for example "
                    f"{realization_input['example_id']} "
                    f"(attempt {attempt + 1}/{MAX_RETRIES + 1})..."
                )

                response = call_ollama(prompt)

                candidate = {
                    "full_story": response["full_story"]
                }

                candidate_validation = validate_story(
                    candidate,
                    realization_input
                )

                output = candidate
                raw_response = response["raw_response"]
                metadata = response["metadata"]

                print("Ollama responded.")
                print(
                    "Validation passed:",
                    candidate_validation["validation_passed"]
                )

                if candidate_validation["issues"]:
                    print("Issues:")
                    for issue in candidate_validation["issues"]:
                        print("-", issue)

                if candidate_validation["warnings"]:
                    print("Warnings:")
                    for warning in candidate_validation["warnings"]:
                        print("-", warning)

                if candidate_validation["validation_passed"]:
                    break

                if RETRY_ON_VALIDATION_FAILURE and attempt < MAX_RETRIES:
                    prompt = (
                        build_prompt(realization_input)
                        + "\n\nRevise the narrative to correct these issues:\n"
                        + "\n".join(
                            f"- {issue}"
                            for issue in candidate_validation["issues"]
                        )
                        + "\nReturn only the corrected narrative text."
                    )
                    time.sleep(RETRY_WAIT_SECONDS)

            except Exception as error:
                generation_error = f"{type(error).__name__}: {error}"
                print("Ollama error:")
                print(generation_error)

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_SECONDS)

        if output is None:
            if not ALLOW_FALLBACK:
                raise RuntimeError(generation_error)

            output = deterministic_fallback(realization_input)
            source = "deterministic_fallback"
            used_fallback = True
        else:
            source = "ollama"
            used_fallback = False

    validation = validate_story(output, realization_input)

    return {
        "example_id": realization_input["example_id"],
        "generation_source": source,
        "model_name": (
            OLLAMA_MODEL
            if source == "ollama"
            else "deterministic_rule_fallback"
        ),
        "used_fallback": used_fallback,
        "generation_error": generation_error,
        "raw_response": raw_response,
        "ollama_metadata": metadata,
        "visible_story_plan": story_plan_map[
            int(realization_input["example_id"])
        ],
        "full_story": output["full_story"],
        "validation": validation,
        "input": {
            "discourse_text": realization_input["discourse_text"],
            "discourse_type": realization_input["discourse_type"]
        },
        "assessment": {
            "reference_label": realization_input["reference_label"],
            "predicted_label": realization_input["predicted_label"],
            "prediction_status": realization_input["prediction_status"]
        },
        "probabilities": realization_input["probabilities"]
    }

# %% [Original notebook cell 9]
# ============================================================
# TEST ONE EXAMPLE
# ============================================================

SELECTED_EXAMPLE_INDEX = 0

test_input = realization_inputs[SELECTED_EXAMPLE_INDEX]
test_result = generate_one_story(test_input)

print("\nEXAMPLE ID        :", test_result["example_id"])
print("GENERATION SOURCE :", test_result["generation_source"])
print("USED FALLBACK     :", test_result["used_fallback"])
print("GENERATION ERROR  :", test_result["generation_error"])
print("VALIDATION        :", test_result["validation"])

print("\nFULL STORY\n")
print(test_result["full_story"])

if test_result["generation_source"] != "ollama":
    print(
        "\nWARNING: Ollama was not used. Read GENERATION ERROR above "
        "before generating all examples."
    )

# %% [Original notebook cell 10]
# ============================================================
# GENERATE ALL EXAMPLES
# ============================================================

all_results = []

for position, realization_input in enumerate(realization_inputs, start=1):
    print(
        f"[{position}/{len(realization_inputs)}] "
        f"Generating example {realization_input['example_id']}..."
    )

    result = generate_one_story(realization_input)
    all_results.append(result)

    print(
        "  source=",
        result["generation_source"],
        "| passed=",
        result["validation"]["validation_passed"],
        "| warnings=",
        len(result["validation"]["warnings"])
    )

print("\nGeneration completed.")
print("Total:", len(all_results))

# %% [Original notebook cell 11]
# ============================================================
# FLATTEN OUTPUTS
# ============================================================

full_story_rows = []
visible_plan_rows = []
validation_rows = []
dashboard_rows = []

for result in all_results:
    example_id = result["example_id"]
    plan = result["visible_story_plan"]
    assessment = result["assessment"]
    probabilities = result["probabilities"]
    validation = result["validation"]

    full_story_rows.append({
        "example_id": example_id,
        "discourse_type": result["input"]["discourse_type"],
        "discourse_text": result["input"]["discourse_text"],
        "reference_label": assessment["reference_label"],
        "predicted_label": assessment["predicted_label"],
        "prediction_status": assessment["prediction_status"],
        "generation_source": result["generation_source"],
        "model_name": result["model_name"],
        "used_fallback": result["used_fallback"],
        "full_story": result["full_story"]
    })

    visible_plan_rows.append({
        "example_id": example_id,
        "hook_type": plan["hook"]["type"],
        "hook_content": plan["hook"]["content"],
        "hero_entity": plan["hero"]["entity"],
        "hero_goal": plan["hero"]["goal"],
        "sidekick_entity": plan["sidekick"]["entity"],
        "sidekick_role": plan["sidekick"]["role"],
        "conflict": plan["conflict"]["central_question"],
        "climax": plan["climax"]["content"],
        "resolution": plan["resolution"]["final_assessment"],
        "improvement_target": plan["resolution"]["improvement_target"],
        "narrative_order": " → ".join(plan["narrative_order"]),
        "engine_controlled": plan["engine_controlled"],
        "llm_selected_structure": plan["llm_selected_structure"]
    })

    validation_rows.append({
        "example_id": example_id,
        "generation_source": result["generation_source"],
        "used_fallback": result["used_fallback"],
        "validation_passed": validation["validation_passed"],
        "issues": stringify_list(validation["issues"]),
        "warnings": stringify_list(validation["warnings"]),
        "word_count": validation["word_count"],
        "character_count": validation["character_count"],
        "generation_error": result["generation_error"]
    })

    dashboard_rows.append({
        "example_id": example_id,
        "discourse_type": result["input"]["discourse_type"],
        "discourse_text": result["input"]["discourse_text"],
        "reference_label": assessment["reference_label"],
        "predicted_label": assessment["predicted_label"],
        "prediction_status": assessment["prediction_status"],
        "prob_ineffective": probabilities["Ineffective"],
        "prob_adequate": probabilities["Adequate"],
        "prob_effective": probabilities["Effective"],
        "hook_type": plan["hook"]["type"],
        "hook_content": plan["hook"]["content"],
        "hero_entity": plan["hero"]["entity"],
        "hero_goal": plan["hero"]["goal"],
        "sidekick_entity": plan["sidekick"]["entity"],
        "sidekick_role": plan["sidekick"]["role"],
        "conflict": plan["conflict"]["central_question"],
        "climax": plan["climax"]["content"],
        "improvement_target": plan["resolution"]["improvement_target"],
        "full_story": result["full_story"],
        "generation_source": result["generation_source"],
        "validation_passed": validation["validation_passed"],
        "validation_issues": stringify_list(validation["issues"]),
        "validation_warnings": stringify_list(validation["warnings"]),
        "engine_controlled": plan["engine_controlled"],
        "llm_selected_structure": plan["llm_selected_structure"]
    })

full_stories_df = pd.DataFrame(full_story_rows)
visible_plans_df = pd.DataFrame(visible_plan_rows)
validation_df = pd.DataFrame(validation_rows)
dashboard_df = pd.DataFrame(dashboard_rows)

print(full_stories_df.head())
print(validation_df.head())

# %% [Original notebook cell 12]
# ============================================================
# QUALITY REPORT
# ============================================================

quality_report = {
    "number_of_examples": len(all_results),
    "ollama_generated": int(
        sum(x["generation_source"] == "ollama" for x in all_results)
    ),
    "fallback_generated": int(
        sum(
            x["generation_source"] == "deterministic_fallback"
            for x in all_results
        )
    ),
    "validation_passed": int(
        validation_df["validation_passed"].sum()
    ),
    "validation_failed": int(
        (~validation_df["validation_passed"]).sum()
    ),
    "average_word_count": float(validation_df["word_count"].mean()),
    "visible_story_plan_exported": True,
    "llm_output_schema": {"full_story": "string"},
    "scientific_note": (
        "The LLM produces only the final narrative. "
        "The complete story plan remains engine-controlled and is merged "
        "with the narrative after generation."
    )
}

print(json.dumps(quality_report, indent=2, ensure_ascii=False))

# %% [Original notebook cell 13]


# %% [Original notebook cell 14]
# ============================================================
# EXPORT
# ============================================================

with open(FINAL_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

with open(QUALITY_REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(quality_report, f, indent=2, ensure_ascii=False)

full_stories_df.to_csv(FULL_STORIES_CSV_PATH, index=False, encoding="utf-8-sig")
visible_plans_df.to_csv(VISIBLE_PLANS_CSV_PATH, index=False, encoding="utf-8-sig")
validation_df.to_csv(VALIDATION_CSV_PATH, index=False, encoding="utf-8-sig")
dashboard_df.to_csv(DASHBOARD_CSV_PATH, index=False, encoding="utf-8-sig")

with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
    dashboard_df.to_excel(writer, sheet_name="Dashboard_Ready", index=False)
    full_stories_df.to_excel(writer, sheet_name="Full_Stories", index=False)
    visible_plans_df.to_excel(writer, sheet_name="Visible_Story_Plans", index=False)
    validation_df.to_excel(writer, sheet_name="Validation", index=False)

print("\nSTEP 3 V2 COMPLETED")
print("-", DASHBOARD_CSV_PATH)
print("-", FINAL_JSON_PATH)
print("-", EXCEL_PATH)
