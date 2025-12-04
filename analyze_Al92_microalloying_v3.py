import pandas as pd
import joblib
import numpy as np

# -------------------------------------------------------------------
# Paths (adjust if needed)
# -------------------------------------------------------------------
MODEL_PATH      = "models_cfg10b_highEI_softWeights_v4.joblib"
MICROALLOY_CSV  = "Al92_microalloying_MLReady.csv"  # ML-ready with all features
OUT_CSV         = "Al92_microalloying_with_predictions.csv"

# -------------------------------------------------------------------
# 1. Load model dict and unpack structure
# -------------------------------------------------------------------
print("Loading model dict...")
model_dict = joblib.load(MODEL_PATH)
print("Top-level keys:", model_dict.keys())   # ['YS', 'UTS', 'Elongation']

def unpack_target(target_key: str):
    """Return (model, imputer, features) for a given target ('YS','UTS','Elongation')."""
    entry = model_dict[target_key]
    if not isinstance(entry, dict):
        raise TypeError(f"model_dict['{target_key}'] is {type(entry)}, expected dict with keys ['model','imputer','features'].")

    for needed in ["model", "imputer", "features"]:
        if needed not in entry:
            raise KeyError(f"model_dict['{target_key}'] is missing key '{needed}'. Has keys: {list(entry.keys())}")

    model    = entry["model"]
    imputer  = entry["imputer"]
    features = entry["features"]

    if not hasattr(model, "predict"):
        raise TypeError(f"model_dict['{target_key}']['model'] has no .predict; type={type(model)}")

    return model, imputer, features

ys_model, ys_imputer, ys_features = unpack_target("YS")
uts_model, uts_imputer, uts_features = unpack_target("UTS")
el_model, el_imputer, el_features = unpack_target("Elongation")

print("\nFeature counts:")
print("  YS features:", len(ys_features))
print("  UTS features:", len(uts_features))
print("  Elongation features:", len(el_features))

# -------------------------------------------------------------------
# 2. Load microalloying ML-ready data
# -------------------------------------------------------------------
print("\nLoading microalloying ML-ready data...")
df = pd.read_csv(MICROALLOY_CSV)
print("Microalloying dataframe shape:", df.shape)

# Sanity check: make sure we have at least the union of all feature columns
all_features = sorted(set(ys_features) | set(uts_features) | set(el_features))
print("Total unique features across all targets:", len(all_features))

# Create any missing feature columns with 0.0
for col in all_features:
    if col not in df.columns:
        df[col] = 0.0

# -------------------------------------------------------------------
# 3. Predict YS / UTS / Elongation
# -------------------------------------------------------------------
print("\nPredicting [YS, UTS, Elongation]...")

# YS
X_ys = df[ys_features].copy()
X_ys_imp = ys_imputer.transform(X_ys)
df["YS"] = ys_model.predict(X_ys_imp)

# UTS
X_uts = df[uts_features].copy()
X_uts_imp = uts_imputer.transform(X_uts)
df["UTS"] = uts_model.predict(X_uts_imp)

# Elongation
X_el = df[el_features].copy()
X_el_imp = el_imputer.transform(X_el)
df["Elongation"] = el_model.predict(X_el_imp)

# -------------------------------------------------------------------
# 4. Compute Q (use your real definition!)
# -------------------------------------------------------------------
def compute_Q(ys, uts, elong):
    # TODO: replace this with your actual Q definition from the paper
    return uts + 30.0 * elong

df["Q"] = compute_Q(df["YS"], df["UTS"], df["Elongation"])

df.to_csv(OUT_CSV, index=False)
print(f"Saved predictions to {OUT_CSV} (shape: {df.shape})")

# -------------------------------------------------------------------
# 5. Baseline (Delta_atpct == 0 → Al92Ti2Fe2Co2Ni2, no microalloy)
# -------------------------------------------------------------------
if "Delta_atpct" not in df.columns:
    raise ValueError("Delta_atpct column not found in microalloying CSV.")

if "MicroElement" not in df.columns:
    raise ValueError("MicroElement column not found in microalloying CSV.")

base = df[df["Delta_atpct"] == 0.0].copy()

print("\nBaseline rows (Delta_atpct = 0):")
cols_show = ["MicroElement","Delta_atpct","YS","UTS","Elongation","Q"]
cols_show = [c for c in cols_show if c in base.columns]
print(base[cols_show])

base_mean = base[["YS","UTS","Elongation","Q"]].mean()
print("\nAverage baseline over all baseline rows:")
print(base_mean)

# -------------------------------------------------------------------
# 6. Summaries Δ vs baseline
# -------------------------------------------------------------------
summary_rows = []

for elem, df_elem in df.groupby("MicroElement"):
    for delta, df_level in df_elem.groupby("Delta_atpct"):
        ys_mean  = df_level["YS"].mean()
        uts_mean = df_level["UTS"].mean()
        e_mean   = df_level["Elongation"].mean()
        q_mean   = df_level["Q"].mean()

        summary_rows.append({
            "MicroElement": elem,
            "Delta_atpct": delta,
            "YS_mean": ys_mean,
            "UTS_mean": uts_mean,
            "Elong_mean": e_mean,
            "Q_mean": q_mean,
            "dYS": ys_mean - base_mean["YS"],
            "dUTS": uts_mean - base_mean["UTS"],
            "dE":  e_mean  - base_mean["Elongation"],
            "dQ":  q_mean  - base_mean["Q"],
        })

summary = pd.DataFrame(summary_rows)

print("\n=== Δ vs baseline (positive = improvement) ===")
print(summary.sort_values(["Delta_atpct","MicroElement"]))

# -------------------------------------------------------------------
# 7. Ranking for Delta_atpct = 1%
# -------------------------------------------------------------------
delta_target = 1.0

rank_dQ = summary[summary["Delta_atpct"] == delta_target] \
            .sort_values("dQ", ascending=False)

print(f"\n=== Ranking at Delta_atpct = {delta_target} (sorted by ΔQ) ===")
print(rank_dQ[["MicroElement","Delta_atpct","YS_mean","UTS_mean",
               "Elong_mean","Q_mean","dE","dQ"]])

rank_dE = summary[summary["Delta_atpct"] == delta_target] \
            .sort_values("dE", ascending=False)

print(f"\n=== Ranking at Delta_atpct = {delta_target} (sorted by ΔElongation) ===")
print(rank_dE[["MicroElement","Delta_atpct","YS_mean","UTS_mean",
               "Elong_mean","Q_mean","dE","dQ"]])
