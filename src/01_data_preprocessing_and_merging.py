"""Data preprocessing and dataset integration.

Converted from the original research notebook while preserving the original code-cell execution order.
"""


# %% [Original notebook cell 1]
import pandas as pd
import numpy as np
from pathlib import Path

# Project-relative paths. Put the original PERSUADE files in data/raw/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# %% [Original notebook cell 2]
path = RAW_DIR / "persuade_corpus_2.0_test.csv"
df = pd.read_csv(path, low_memory=False)

# %% [Original notebook cell 3]
# Check how many Unannotated rows first
print("\nDiscourse type distribution (before):")
print(df['discourse_type'].value_counts(dropna=False))

# --------------------------------------------------
# Remove Unannotated rows
# --------------------------------------------------
df_clean = df[df['discourse_type'] != 'Unannotated'].copy()

# Reset index (important)
df_clean.reset_index(drop=True, inplace=True)
#-------------------------------------------------
# Check how many effectiveness rows first
#-------------------------------------------------
print("\nEfectiveness distribution (before):")
print(df['discourse_effectiveness'].value_counts(dropna=False))
# --------------------------------------------------
# Remove Adequete rows
# --------------------------------------------------
df_clean = df_clean[df_clean['discourse_effectiveness'] != 'Adequate'].copy()
df_clean = df_clean.dropna(subset=['discourse_effectiveness'])
# Reset index (important)
df_clean.reset_index(drop=True, inplace=True)
# --------------------------------------------------
# Check result
# --------------------------------------------------
print("\nAfter:", df_clean.shape)

print("\nDiscourse type distribution (after):")
print(df_clean['discourse_type'].value_counts(dropna=False))
print("\nEfectiveness distribution (after):")
print(df_clean['discourse_effectiveness'].value_counts(dropna=False))

# %% [Original notebook cell 4]
output_path = PROCESSED_DIR / "persuade_test_clean_no_unannotated.csv"
df_clean.to_csv(output_path, index=False)

print("Saved cleaned dataset to:", output_path)

# %% [Original notebook cell 5]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1) Load data
# =========================
path = PROCESSED_DIR / "persuade_test_clean_no_unannotated.csv"
df = pd.read_csv(path, low_memory=False)

# =========================
# 2) Convert hidden missing values to NaN
# =========================
# Handles empty strings and strings made only of spaces
df = df.replace(r'^\s*$', np.nan, regex=True)

# =========================
# 3) Drop unnecessary columns
# =========================
drop_cols = [
    'essay_id',
    'essay_id_comp',
    'competition_set',
    'discourse_id',
    'hierarchical_id',
    'hierarchical_text',
    'hierarchical_label'
]

# Free-text columns dropped for RF imputation
text_cols_to_drop = [
    'full_text',
    'discourse_text',
    'source_text'
]

df = df.drop(columns=drop_cols + text_cols_to_drop, errors='ignore')

# =========================
# 4) Columns to impute
# =========================
target_columns = [
    'ell_status',
    'grade_level',
    'student_disability_status',
    'economically_disadvantaged'
]

# Keep a copy before imputation for comparison
df_before = df.copy()

# =========================
# 5) Encode categorical columns
#    while preserving NaN
# =========================
df_encoded = df.copy()
encoding_maps = {}
decoding_maps = {}

cat_cols = df_encoded.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

for col in cat_cols:
    df_encoded[col] = df_encoded[col].astype('object')
    
    non_null_values = pd.Series(df_encoded[col].dropna().unique())
    value_to_int = {val: idx for idx, val in enumerate(non_null_values)}
    int_to_value = {idx: val for val, idx in value_to_int.items()}
    
    encoding_maps[col] = value_to_int
    decoding_maps[col] = int_to_value
    
    df_encoded[col] = df_encoded[col].map(value_to_int)

# =========================
# 6) Function to impute one column
#    using important features
# =========================
def impute_with_rf(df_in, target_col, top_n=10, n_estimators=200, random_state=42):
    df_work = df_in.copy()

    print(f"\n===== Imputing: {target_col} =====")

    train_idx = df_work[target_col].notna()
    pred_idx = df_work[target_col].isna()

    print("Missing before:", pred_idx.sum())

    if pred_idx.sum() == 0:
        print(f"No missing values in {target_col}.")
        return df_work

    X_train = df_work.loc[train_idx].drop(columns=[target_col])
    y_train = df_work.loc[train_idx, target_col]

    X_pred = df_work.loc[pred_idx].drop(columns=[target_col])

    # Fill missing values in predictors only
    X_train = X_train.fillna(-1)
    X_pred = X_pred.fillna(-1)

    # Force categorical target to integer labels
    y_train = y_train.astype(int)

    # First model: get feature importance
    rf_full = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf_full.fit(X_train, y_train)

    importances = pd.Series(rf_full.feature_importances_, index=X_train.columns)
    important_features = importances.sort_values(ascending=False).head(top_n).index.tolist()

    print("Top important features:")
    print(important_features)

    # Second model: train only on top important features
    rf_top = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf_top.fit(X_train[important_features], y_train)

    # Predict missing values
    y_pred = rf_top.predict(X_pred[important_features])

    # Fill them back
    df_work.loc[pred_idx, target_col] = y_pred

    print("Missing after:", df_work[target_col].isna().sum())

    return df_work

# =========================
# 7) Check missing values before
# =========================
print("Missing values before imputation:")
print(df_encoded[target_columns].isna().sum())

# =========================
# 8) Impute each target column
# =========================
for col in target_columns:
    df_encoded = impute_with_rf(df_encoded, col, top_n=10)

# =========================
# 9) Check missing values after
# =========================
print("\nMissing values after imputation:")
print(df_encoded[target_columns].isna().sum())

# =========================
# 10) Decode categorical columns back
# =========================
df_imputed = df_encoded.copy()

for col in cat_cols:
    reverse_map = decoding_maps[col]
    df_imputed[col] = df_imputed[col].map(reverse_map)

# Optional: for grade_level if you want it numeric
# df_imputed['grade_level'] = pd.to_numeric(df_imputed['grade_level'], errors='coerce')

# =========================
# 11) Distribution tables
# =========================
for col in target_columns:
    print(f"\n===== {col} =====")
    
    before = df_before[col].value_counts(dropna=False, normalize=True)
    after = df_imputed[col].value_counts(dropna=False, normalize=True)
    
    comparison = pd.DataFrame({
        'Before': before,
        'After': after
    }).fillna(0)
    
    print(comparison)

# =========================
# 12) Distribution plots
# =========================
for col in target_columns:
    compare = pd.DataFrame({
        'Before': df_before[col].value_counts(normalize=True, dropna=False),
        'After': df_imputed[col].value_counts(normalize=True, dropna=False)
    }).fillna(0)

    compare.plot(kind='bar', figsize=(8, 5))
    plt.title(f"{col} Distribution: Before vs After Imputation")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# =========================
# 13)  save
# =========================
output_path = PROCESSED_DIR / "persuade_test_corpus_2.0_imputed.csv"
df_imputed.to_csv(output_path, index=False)
print(f"Saved imputed file to: {output_path}")

# %% [Original notebook cell 6]
# =========================================================
# 14) Reload the original cleaned dataset
#     to recover dropped columns
# =========================================================
path_original = PROCESSED_DIR / "persuade_test_clean_no_unannotated.csv"
df_original = pd.read_csv(path_original, low_memory=False)

# Convert hidden missing values to NaN
df_original = df_original.replace(r'^\s*$', np.nan, regex=True)

# Reset index to make sure rows align
df_original = df_original.reset_index(drop=True)
df_imputed = df_imputed.reset_index(drop=True)

# =========================================================
# 15) Add back the columns you want
# =========================================================
cols_to_add = ['essay_id', 'discourse_text', 'full_text', 'source_text']

for col in cols_to_add:
    df_imputed[col] = df_original[col]

# =========================================================
# 16) Replace missing source_text with "without"
# =========================================================
df_imputed['source_text'] = df_imputed['source_text'].fillna('without')

# =========================================================
# 17) Check result
# =========================================================
print(df_imputed[['essay_id', 'discourse_text', 'full_text', 'source_text']].head())

print("\nMissing values in source_text after replacement:")
print(df_imputed['source_text'].isna().sum())

# =========================================================
# 18) Save final corrected dataset
# =========================================================
output_path = PROCESSED_DIR / "persuade_test_corpus_2.0_imputed_final.csv"
df_imputed.to_csv(output_path, index=False)

print(f"\nSaved final file to: {output_path}")

# %% [Original notebook cell 7]
path = PROCESSED_DIR / "persuade_test_corpus_2.0_imputed_final.csv"
df = pd.read_csv(path, low_memory=False)

# %% [Original notebook cell 8]
print("===== discourse_effectiveness Missing =====")
print(df['discourse_effectiveness'].isna().value_counts())

print("\nPercentage:")
print(df['discourse_effectiveness'].isna().mean() * 100, "%")

# %% [Original notebook cell 9]
# =========================================================
#  Drop rows with missing discourse_effectiveness
# =========================================================
print("Before:", df.shape)

df = df[df['discourse_effectiveness'].notna()].copy()

print("After:", df.shape)

# Check
print("\nRemaining missing values:")
print(df['discourse_effectiveness'].isna().sum())

# =========================================================
# Save final dataset
# =========================================================
output_path = PROCESSED_DIR / "persuade_test_corpus_2.0_imputed_final_version.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved final file to: {output_path}")

# %% [Original notebook cell 10]
path = PROCESSED_DIR / "persuade_test_corpus_2.0_imputed_final_version.csv"
df = pd.read_csv(path, low_memory=False)

# %% [Original notebook cell 11]
df.isnull().mean() * 100

# %% [Original notebook cell 12]
import pandas as pd

path1 = RAW_DIR / "persuade_corpus_2.0_imputed_final_version.csv"
path2 = PROCESSED_DIR / "persuade_test_corpus_2.0_imputed_final_version.csv"

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

# %% [Original notebook cell 13]
print(df1.shape)
print(df2.shape)

# %% [Original notebook cell 14]
persuade_main_corpus_version = pd.concat([df1, df2], ignore_index=True)

print(persuade_main_corpus_version.shape)

# %% [Original notebook cell 15]
dup_percent = persuade_main_corpus_version.duplicated().mean() * 100
print(f"Duplicate rows: {dup_percent:.2f}%")

# %% [Original notebook cell 16]
dup_percent_text = persuade_main_corpus_version.duplicated(subset=['full_text']).mean() * 100
print(f"Duplicate texts: {dup_percent_text:.2f}%")
num_dup = persuade_main_corpus_version.duplicated(subset=['full_text']).sum()
print("Number of duplicate texts:", num_dup)
persuade_main_corpus_version['full_text_clean'] = (
    persuade_main_corpus_version['full_text']
    .str.strip()
    .str.lower()
)

dup_percent_clean = (
    persuade_main_corpus_version
    .duplicated(subset=['full_text_clean'])
    .mean() * 100
)

print(f"Duplicate texts (cleaned): {dup_percent_clean:.2f}%")

# %% [Original notebook cell 17]
output_path = PROCESSED_DIR / "persuade_main_corpus_version.csv"

persuade_main_corpus_version.to_csv(output_path, index=False)

# %% [Original notebook cell 18]
path= PROCESSED_DIR / "persuade_main_corpus_version.csv"
df = pd.read_csv(path, low_memory=False)
print("\nEfectiveness distribution (before):")
print(df['discourse_effectiveness'].value_counts(dropna=False))

# %% [Original notebook cell 19]
path= PROCESSED_DIR / "persuade_main_corpus_version.csv"
df = pd.read_csv(path, low_memory=False)

# %% [Original notebook cell 20]
# Print all columns (variables) in your dataset
print("Dataset Variables / Columns:\n")
for col in df.columns:
    print(col)

# Show data types and non-null counts
print("\n\nDataset Attributes (info):\n")
df.info()

# Detailed summary for all columns
print("\n\nDetailed Description:\n")
print(df.describe(include='all').transpose())

# Optional: Print unique values count for each variable
print("\n\nUnique Values Per Variable:\n")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# %% [Original notebook cell 21]
# Print each variable with all its unique values

for col in df.columns:
    print(f"\\n{'='*60}")
    print(f"Variable: {col}")
    print(f"Number of Unique Values: {df[col].nunique()}")
    print("Unique Values:")
    print(df[col].dropna().unique())

# %% [Original notebook cell 22]

# %% [Original notebook cell 23]
selected_columns = [
    'holistic_essay_score',
    'discourse_start',
    'discourse_end',
    'discourse_type',
    'discourse_type_num',
    'discourse_effectiveness',
    'provider',
    'task',
    'prompt_name',
    'assignment',
    'gender',
    'grade_level',
    'ell_status',
    'race_ethnicity',
    'economically_disadvantaged',
    'student_disability_status',
    'essay_word_count',
    'essay_id',
    'discourse_text',
    'full_text',
    'source_text'
]

for col in selected_columns:
    print("\n" + "="*80)
    print(f"VARIABLE: {col}")
    print("="*80)

    print("\nMissing values:")
    print(df[col].isna().sum())

    print("\nNumber of unique values:")
    print(df[col].nunique(dropna=False))

    print("\nDistribution / value counts:")
    print(df[col].value_counts(dropna=False).head(30))

    if df[col].dtype == "object":
        print("\nText length distribution:")
        lengths = df[col].dropna().astype(str).str.len()
        print(lengths.describe())
    else:
        print("\nNumeric distribution:")
        print(df[col].describe())

# %% [Original notebook cell 24]

