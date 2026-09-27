"""LIME and Integrated Gradients explanations for discourse effectiveness.

Converted from the original research notebook while preserving the original code-cell execution order.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "deberta_saved_model"
DATA_PATH = PROJECT_ROOT / "outputs" / "deberta_test_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xai_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% [Original notebook cell 1]
import sys

# %% [Original notebook cell 2]
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for notebooks too
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
from captum.attr import LayerIntegratedGradients

print("All imports OK")

# %% [Original notebook cell 3]
# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── output files for this notebook version ───────────────────────────────────
SELECTED_EXAMPLES_CSV = OUTPUT_DIR / "xai_copy_selected_examples.csv"
SELECTED_EXAMPLES_XLSX = OUTPUT_DIR / "xai_copy_selected_examples.xlsx"
SUMMARY_CSV = OUTPUT_DIR / "xai_copy_summary.csv"
TOKEN_WEIGHTS_XLSX = OUTPUT_DIR / "xai_copy_token_weights.xlsx"

# ── labels ────────────────────────────────────────────────────────────────────
LABELS   = ["Ineffective", "Adequate", "Effective"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

# ── PERSUADE corpus discourse types ──────────────────────────────────────────
DISCOURSE_TYPES = [
    "Claim",
    "Evidence",
    "Position",
    "Counterclaim",
    "Rebuttal",
    "Concluding Statement",
    "Lead",
]

# ── XAI parameters ────────────────────────────────────────────────────────────
LIME_NUM_FEATURES = 15
LIME_NUM_SAMPLES  = 500    # increase to 1000 for more stable results
IG_N_STEPS        = 50     # 50 to avoid OOM — internal_batch_size=4 handles chunking
IG_MAX_LEN        = 128

print("Configuration set.")
print(f"Output directory : {OUTPUT_DIR}")
print(f"Discourse types  : {DISCOURSE_TYPES}")

# %% [Original notebook cell 4]
print(f"GPUs available: {torch.cuda.device_count()}")
PRIMARY_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── single-GPU model (LIME + IG) ─────────────────────────────────────────────
tokenizer      = AutoTokenizer.from_pretrained(MODEL_PATH)
_model_single  = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
_model_single.to(PRIMARY_DEVICE)
_model_single.eval()

# ── DataParallel model (batch inference) ─────────────────────────────────────
if torch.cuda.device_count() > 1:
    model_parallel = torch.nn.DataParallel(
        AutoModelForSequenceClassification.from_pretrained(MODEL_PATH),
        device_ids=list(range(torch.cuda.device_count()))
    )
    model_parallel.to(PRIMARY_DEVICE)
    model_parallel.eval()
    print(f"DataParallel enabled on GPUs: {list(range(torch.cuda.device_count()))}")
else:
    model_parallel = _model_single
    print("Single GPU / CPU mode — model_parallel = _model_single")

print(f"Primary device: {PRIMARY_DEVICE}")

# %% [Original notebook cell 5]
def predict_proba_batch(discourse_types, discourse_texts, batch_size=32):
    """
    Returns (N, 3) softmax probabilities.
    Uses DataParallel — splits batches across all 4 GPUs automatically.
    batch_size=32 with 4 GPUs = 8 samples per GPU per forward pass.
    """
    all_probs = []
    for i in range(0, len(discourse_texts), batch_size):
        dt  = list(discourse_types[i:i+batch_size])
        txt = list(discourse_texts[i:i+batch_size])
        enc = tokenizer(dt, txt, truncation=True, padding=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(PRIMARY_DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model_parallel(**enc).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


def predict_single(discourse_type, discourse_text):
    """Single-sample prediction on primary GPU — used by LIME and IG."""
    enc = tokenizer(discourse_type, discourse_text, truncation=True,
                    padding=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(PRIMARY_DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = _model_single(**enc).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()[0]

print("Prediction helpers ready.")

# %% [Original notebook cell 6]
test_df = pd.read_csv(DATA_PATH)
print(f"Test set: {len(test_df)} rows")
print(f"Discourse types in data: {sorted(test_df['discourse_type'].unique())}")
print(f"Class distribution:")
print(test_df['true_label_name'].value_counts())

# %% [Original notebook cell 7]
# ============================================================
# SECTION 5 — SELECT XAI EXAMPLES
# ============================================================
# Goal:
# - Keep the same 21-example output structure:
#   [01/21] Claim / Ineffective ...
#   [02/21] Claim / Adequate ...
#   ...
# - Force the 3 thesis examples to be included.
# - Randomly select only the remaining combinations.
# - Save the final selection under xai_copy_* names.

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1) Fixed thesis examples that MUST be included
# ---------------------------------------------------------------------
FIXED_TEXTS = [
    {
        "name": "Fixed Claim Example",
        "expected_discourse_type": "Claim",
        "discourse_text": "Online classes allow for a lack of personal interaction between students and teachers"
    },
    {
        "name": "Fixed Evidence Example",
        "expected_discourse_type": "Evidence",
        "discourse_text": "For example; there are days where I just feel weak and tired but I wouldn't want to miss a review day for a major test the next class. If there was online school, I would be able to connect with my teacher and the classes from home where I feel comfortable. Also, my mom wouldn't have to worry that i'm failing my classes because i'm not showing up to them. I believe that other students would agree with me on this argument because almost everyone I know, wishes they could attend school from home."
    },
    {
        "name": "Fixed Concluding Statement Example",
        "expected_discourse_type": "Concluding Statement",
        "discourse_text": "On the contrary to what is beleived to be, the author provided good evidence as to exciting new ways to hopefully one day be able to learn more about Venus, and understand our solar system at the very least by one planet more."
    }
]

def normalize_text_for_matching(x):
    """Normalize spaces and case so small spacing differences do not break matching."""
    return " ".join(str(x).split()).strip().lower()

test_df = test_df.copy()
test_df["_norm_text"] = test_df["discourse_text"].apply(normalize_text_for_matching)

# ---------------------------------------------------------------------
# 2) Find the 3 fixed examples in test_df
# ---------------------------------------------------------------------
fixed_rows = []

for item in FIXED_TEXTS:
    target_norm = normalize_text_for_matching(item["discourse_text"])

    # Exact normalized match
    subset = test_df[
        (test_df["_norm_text"] == target_norm) &
        (test_df["discourse_type"] == item["expected_discourse_type"])
    ]

    # Fallback: contains-based match using the first 60 characters
    if len(subset) == 0:
        short_key = target_norm[:60]
        subset = test_df[
            (test_df["_norm_text"].str.contains(short_key, regex=False, na=False)) &
            (test_df["discourse_type"] == item["expected_discourse_type"])
        ]

    if len(subset) == 0:
        print(f"WARNING: Fixed example not found: {item['name']}")
        continue

    row = subset.iloc[0].copy()
    row["_fixed_name"] = item["name"]
    fixed_rows.append(row)

print(f"Fixed examples found: {len(fixed_rows)}/{len(FIXED_TEXTS)}")

for row in fixed_rows:
    print(
        f"  - {row['_fixed_name']} | "
        f"{row['discourse_type']} / {row['true_label_name']} | "
        f"index={row.name}"
    )

# ---------------------------------------------------------------------
# 3) Convert fixed examples into a dictionary by (discourse_type, true_label)
# ---------------------------------------------------------------------
# This lets each fixed example replace the random example in the correct
# discourse_type × class position.
fixed_by_combo = {}

for row in fixed_rows:
    key = (row["discourse_type"], row["true_label_name"])
    fixed_by_combo[key] = row

# ---------------------------------------------------------------------
# 4) Build the final 21-example selection
# ---------------------------------------------------------------------
samples = []
missing = []
used_fixed_indices = set()

for dtype in DISCOURSE_TYPES:
    for label_name in LABELS:

        key = (dtype, label_name)

        # If this combination has one of the fixed thesis examples, use it.
        if key in fixed_by_combo:
            row = fixed_by_combo[key]
            selection_source = "FIXED"
            used_fixed_indices.add(row.name)

        # Otherwise, randomly sample one example from the same combination.
        else:
            subset = test_df[
                (test_df["discourse_type"] == dtype) &
                (test_df["true_label_name"] == label_name) &
                (~test_df.index.isin(used_fixed_indices))
            ]

            if len(subset) == 0:
                missing.append(f"{dtype} / {label_name}")
                continue

            row = subset.sample(1, random_state=RANDOM_STATE).iloc[0]
            selection_source = "RANDOM"

        samples.append({
            "example_id"       : len(samples) + 1,
            "selection_source" : selection_source,
            "idx"              : row.name,
            "discourse_type"   : row["discourse_type"],
            "true_label"       : row["true_label_name"],
            "discourse_text"   : row["discourse_text"],
        })

selected_examples = pd.DataFrame(samples)

print(f"\nSamples selected : {len(selected_examples)}")
if missing:
    print(f"Missing combos   : {missing}")

# ---------------------------------------------------------------------
# 5) Save selected examples permanently
# ---------------------------------------------------------------------
selected_examples.to_csv(SELECTED_EXAMPLES_CSV, index=False)
try:
    selected_examples.to_excel(SELECTED_EXAMPLES_XLSX, index=False)
    print(f"Selected examples Excel path: {SELECTED_EXAMPLES_XLSX}")
except ModuleNotFoundError:
    print("Excel export skipped for selected examples because openpyxl is not installed.")
    print("CSV was saved correctly, so you can continue.")
except Exception as e:
    print(f"Excel export skipped for selected examples: {e}")

print(f"\nSaved selected examples CSV  : {SELECTED_EXAMPLES_CSV}")
print(f"Selected examples Excel path: {SELECTED_EXAMPLES_XLSX}")

# Preview the selection grid
grid = selected_examples[["example_id", "selection_source", "discourse_type", "true_label", "discourse_text"]].copy()
grid["text_preview"] = grid["discourse_text"].str[:90] + "..."
grid = grid.drop(columns="discourse_text")
grid

# %% [Original notebook cell 8]
# random_state goes on the explainer — NOT on explain_instance()
lime_explainer = LimeTextExplainer(class_names=LABELS, random_state=42)

# Module-level variable so lime_predict_fn can read the current discourse_type
_LIME_DTYPE = "Claim"

def lime_predict_fn(texts):
    """
    LIME calls this with 500 perturbed discourse_text strings.
    Uses DataParallel batch inference — splits across all 4 GPUs.
    """
    return predict_proba_batch([_LIME_DTYPE] * len(texts), texts, batch_size=32)

print("LIME explainer ready.")

# %% [Original notebook cell 9]
lig = LayerIntegratedGradients(
    forward_func=lambda input_ids, attention_mask, token_type_ids=None: _model_single(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    ).logits,
    layer=_model_single.deberta.embeddings.word_embeddings
)


def get_ig_attributions(discourse_type, discourse_text, target_label_id):
    """
    Returns (tokens, attributions, convergence_delta).
    Baseline: [MASK] token — correct for DeBERTa disentangled embeddings.
    n_steps=50 to avoid OOM — internal_batch_size=4 processes 4 steps at a time.
    Runs on single GPU (_model_single) — IG requires gradient tracking.
    """
    enc = tokenizer(
        discourse_type, discourse_text,
        truncation=True, padding="max_length",
        max_length=IG_MAX_LEN, return_tensors="pt"
    )
    input_ids      = enc["input_ids"].to(PRIMARY_DEVICE)
    attention_mask = enc["attention_mask"].to(PRIMARY_DEVICE)
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(PRIMARY_DEVICE)

    baseline = torch.full_like(input_ids, tokenizer.mask_token_id)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline,
        additional_forward_args=(attention_mask, token_type_ids),
        target=target_label_id,
        n_steps=100,               # reduced from 200 — prevents OOM on 4GB GPU
        internal_batch_size=4,    # processes 4 interpolation steps at a time
        return_convergence_delta=True
    )
    attr_sum = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    tokens   = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return tokens, attr_sum, float(delta)


def comprehensiveness(discourse_type, text, tok_clean, att_clean, pred_id, k=5):
    """Mask top-k IG tokens and measure confidence drop."""
    orig_prob = predict_single(discourse_type, text)[pred_id]
    ranked    = sorted([(t, a) for t, a in zip(tok_clean, att_clean) if a > 0],
                       key=lambda x: -x[1])[:k]
    masked    = text
    for tok, _ in ranked:
        surface = tok.replace("##", "").replace("\u2581", "")
        masked  = masked.replace(surface, "[MASK]", 1)
    masked_prob = predict_single(discourse_type, masked)[pred_id]
    return float(orig_prob - masked_prob)


print("Integrated Gradients ready.")

# %% [Original notebook cell 10]
# ============================================================
# SECTION 8 — MAIN SYSTEMATIC XAI ANALYSIS LOOP
# ============================================================
# This loop processes all selected examples.
#
# For each example, it saves:
# - prediction probabilities
# - LIME chart and HTML
# - IG chart
# - summary row
# - LIME token weights for all three classes
# - IG token weights for the predicted class
#
# The Excel file xai_copy_token_weights.xlsx is created later from all_token_weights.

results = []
all_token_weights = []

special = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"}

for i, sample in selected_examples.iterrows():

    example_id = int(sample["example_id"])
    dtype      = sample["discourse_type"]
    true_label = sample["true_label"]
    text       = sample["discourse_text"]
    source     = sample["selection_source"]

    print(f"[{example_id:02d}/{len(selected_examples)}] {dtype:22s} / {true_label:12s}", end="  ")

    # ── prediction ───────────────────────────────────────────────────────────
    probs      = predict_single(dtype, text)
    pred_id    = int(np.argmax(probs))
    pred_label = id2label[pred_id]
    correct    = (pred_label == true_label)
    status     = "✓" if correct else "✗ MISCLASSIFIED"
    print(f"Pred: {pred_label:12s} {status}  ({source})")

    # ── LIME ─────────────────────────────────────────────────────────────────
    _LIME_DTYPE = dtype

    lime_exp = lime_explainer.explain_instance(
        text,
        lime_predict_fn,
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
        labels=[0, 1, 2]
    )

    lime_top5 = [w for w, _ in lime_exp.as_list(label=pred_id)[:5]]

    # Save LIME token weights for all three classes
    for class_id, class_name in enumerate(LABELS):
        for token, weight in lime_exp.as_list(label=class_id):
            all_token_weights.append({
                "example_id"       : example_id,
                "selection_source" : source,
                "discourse_type"   : dtype,
                "true_label"       : true_label,
                "predicted_label"  : pred_label,
                "method"           : "LIME",
                "target_class"     : class_name,
                "token"            : token,
                "weight"           : float(weight),
                "weight_normalized": np.nan,
                "discourse_text"   : text
            })

    # LIME bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"LIME  |  Example {example_id:02d}  |  {dtype}  |  True: {true_label}  |  Pred: {pred_label}",
        fontsize=12
    )

    for ax, (lid, lname) in zip(axes, zip([0, 1, 2], LABELS)):
        feats  = lime_exp.as_list(label=lid)
        words  = [f[0] for f in feats]
        scores = [f[1] for f in feats]
        colors = ["#d62728" if s < 0 else "#2ca02c" for s in scores]
        ax.barh(words[::-1], scores[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Class: {lname}")
        ax.set_xlabel("LIME weight")

    plt.tight_layout()
    safe_dtype = dtype.replace(" ", "_")
    safe_true  = true_label.replace(" ", "_")

    lime_png = os.path.join(
        OUTPUT_DIR,
        f"xai_copy_lime_{example_id:02d}_{safe_dtype}_{safe_true}.png"
    )
    plt.savefig(lime_png, dpi=120, bbox_inches="tight")
    plt.close()

    lime_html = os.path.join(
        OUTPUT_DIR,
        f"xai_copy_lime_{example_id:02d}_{safe_dtype}_{safe_true}.html"
    )
    lime_exp.save_to_file(lime_html)

    # ── free GPU memory before IG ────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Integrated Gradients ─────────────────────────────────────────────────
    tokens_ig, attrs_ig, delta_ig = get_ig_attributions(dtype, text, pred_id)

    tok_clean = [t for t in tokens_ig if t not in special]
    att_clean = np.array([a for t, a in zip(tokens_ig, attrs_ig) if t not in special])
    att_norm  = att_clean / (np.abs(att_clean).max() + 1e-9)

    # Save IG token weights for the predicted class
    for token, raw_weight, norm_weight in zip(tok_clean, att_clean, att_norm):
        all_token_weights.append({
            "example_id"       : example_id,
            "selection_source" : source,
            "discourse_type"   : dtype,
            "true_label"       : true_label,
            "predicted_label"  : pred_label,
            "method"           : "IG",
            "target_class"     : pred_label,
            "token"            : token,
            "weight"           : float(raw_weight),
            "weight_normalized": float(norm_weight),
            "discourse_text"   : text
        })

    colors_ig = ["#d62728" if v < 0 else "#2ca02c" for v in att_norm]

    fig, ax = plt.subplots(figsize=(14, max(3, len(tok_clean) // 7)))
    ax.barh(tok_clean[::-1], att_norm[::-1], color=colors_ig[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"IG  |  Example {example_id:02d}  |  {dtype}  |  True: {true_label}  |  Pred: {pred_label}  |  delta={delta_ig:.3f}"
    )
    ax.set_xlabel("Normalised attribution (positive = supports predicted class)")

    plt.tight_layout()
    ig_png = os.path.join(
        OUTPUT_DIR,
        f"xai_copy_ig_{example_id:02d}_{safe_dtype}_{safe_true}.png"
    )
    plt.savefig(ig_png, dpi=120, bbox_inches="tight")
    plt.close()

    comp_drop = comprehensiveness(dtype, text, tok_clean, att_clean, pred_id)

    # ── free GPU memory at end of iteration ──────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── store summary result ─────────────────────────────────────────────────
    results.append({
        "example_id"            : example_id,
        "selection_source"      : source,
        "idx"                   : sample["idx"],
        "discourse_type"        : dtype,
        "true_label"            : true_label,
        "pred_label"            : pred_label,
        "correct"               : correct,
        "prob_ineffective"      : round(float(probs[0]), 4),
        "prob_adequate"         : round(float(probs[1]), 4),
        "prob_effective"        : round(float(probs[2]), 4),
        "lime_top5_tokens"      : ", ".join(lime_top5),
        "ig_convergence_delta"  : round(float(delta_ig), 4),
        "ig_comprehensiveness"  : round(float(comp_drop), 4),
        "lime_png"              : lime_png,
        "lime_html"             : lime_html,
        "ig_png"                : ig_png,
        "discourse_text"        : text
    })

print("\nLoop complete.")
print(f"Total token weights collected: {len(all_token_weights):,}")

# %% [Original notebook cell 11]
# ============================================================
# SECTION 10 — SAVE RESULTS, EXCEL REPORTS & SUMMARY
# ============================================================

# 1) Save summary results
results_df = pd.DataFrame(results)
results_df.to_csv(SUMMARY_CSV, index=False)

print(f"Summary CSV saved: {SUMMARY_CSV}")

# 2) Save all LIME + IG token weights for all examples
token_weights_df = pd.DataFrame(all_token_weights)
TOKEN_WEIGHTS_CSV = TOKEN_WEIGHTS_XLSX.replace(".xlsx", ".csv")
token_weights_df.to_csv(TOKEN_WEIGHTS_CSV, index=False)

try:
    token_weights_df.to_excel(TOKEN_WEIGHTS_XLSX, index=False)
    print(f"Token weights Excel saved: {TOKEN_WEIGHTS_XLSX}")
except ModuleNotFoundError:
    print("Excel export skipped because openpyxl is not installed.")
    print("Token weights were saved as CSV instead.")
except Exception as e:
    print(f"Excel export skipped: {e}")
    print("Token weights were saved as CSV instead.")

print(f"Token weights CSV saved  : {TOKEN_WEIGHTS_CSV}")
print(f"Total token-weight rows  : {len(token_weights_df):,}")

# 3) Display thesis-friendly summary
display_cols = [
    "example_id",
    "selection_source",
    "discourse_type",
    "true_label",
    "pred_label",
    "correct",
    "prob_ineffective",
    "prob_adequate",
    "prob_effective",
    "ig_comprehensiveness",
    "lime_top5_tokens"
]

print(f"\nCorrect predictions : {results_df['correct'].sum()}/{len(results_df)}")
print(f"Mean comprehensiveness drop : {results_df['ig_comprehensiveness'].mean():.4f}")

print("\nFull results:")
display(results_df[display_cols])

print("\nToken weights preview:")
display(token_weights_df.head(20))

# %% [Original notebook cell 12]
VIEW_IDX = 0   # change to 0–20 to browse any sample

row = results_df.iloc[VIEW_IDX]
print(f"discourse_type  : {row['discourse_type']}")
print(f"True label      : {row['true_label']}")
print(f"Predicted label : {row['pred_label']}  {'\u2713' if row['correct'] else '\u2717 MISCLASSIFIED'}")
print(f"Probabilities   : Ineff={row['prob_ineffective']}  Adeq={row['prob_adequate']}  Eff={row['prob_effective']}")
print(f"LIME top-5      : {row['lime_top5_tokens']}")
print(f"IG delta        : {row['ig_convergence_delta']}  (close to 0 = accurate)")
print(f"Comprehensiveness drop : {row['ig_comprehensiveness']:+.4f}  (>0.10 = faithful)")

# %% [Original notebook cell 13]
# show LIME and IG charts side by side
row = results_df.iloc[VIEW_IDX]
fig, axes = plt.subplots(1, 2, figsize=(22, 5))

for ax, path, title in zip(axes,
    [row["lime_png"], row["ig_png"]],
    ["LIME", "Integrated Gradients"]):
    img = plt.imread(path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=13)

plt.tight_layout()
plt.show()
