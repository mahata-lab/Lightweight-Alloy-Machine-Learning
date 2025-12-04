import itertools
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "models_cfg10b_highEI_softWeights_v4.joblib"
PROCESS_CODE = 1000000  # same as training (e.g., LPBF-like)

# ---------------------------
# Load model & unpack targets
# ---------------------------
print("Loading model dict...")
model_dict = joblib.load(MODEL_PATH)
print("Top-level keys:", model_dict.keys())  # ['YS','UTS','Elongation']

def unpack_target(target_key: str):
    """
    Return (model, imputer, features) for a given target ('YS','UTS','Elongation').
    """
    entry = model_dict[target_key]
    if not isinstance(entry, dict):
        raise TypeError(
            f"model_dict['{target_key}'] is {type(entry)}, expected dict with "
            f"keys ['model','imputer','features']."
        )

    for needed in ["model", "imputer", "features"]:
        if needed not in entry:
            raise KeyError(
                f"model_dict['{target_key}'] is missing key '{needed}'. "
                f"Has keys: {list(entry.keys())}"
            )

    model = entry["model"]
    imputer = entry["imputer"]
    features = entry["features"]

    if not hasattr(model, "predict"):
        raise TypeError(
            f"model_dict['{target_key}']['model'] has no .predict; "
            f"type={type(model)}"
        )

    return model, imputer, features

ys_model, ys_imputer, ys_features = unpack_target("YS")
uts_model, uts_imputer, uts_features = unpack_target("UTS")
el_model, el_imputer, el_features = unpack_target("Elongation")

print("\nFeature counts:")
print("  YS features:", len(ys_features))
print("  UTS features:", len(uts_features))
print("  Elongation features:", len(el_features))

# Union of all feature names
all_features = sorted(set(ys_features) | set(uts_features) | set(el_features))
print("Total unique features across all targets:", len(all_features))


# ---------------------------
# Q definition (edit if needed)
# ---------------------------
def compute_Q(ys, uts, elong):
    # Replace with your exact Q from the paper if different
    return uts + 30.0 * elong


# ---------------------------
# Case 1: Add Sc/Zr/Cr at expense of Al, TM fixed at 8 at.%
# ---------------------------
def build_case1_designspace():
    """
    Base: Ti = 2, Fe = 2, Co = 2, Ni = 2 (TM_sum=8).
    Add Sc, Zr, Cr s.t. total_dop <= max_dop, Al = 100 - 8 - total_dop.
    """
    BASE_TM = {"Ti": 2.0, "Fe": 2.0, "Co": 2.0, "Ni": 2.0}
    dopants = ["Sc", "Zr", "Cr"]
    step = 0.5
    max_total_dop = 4.0  # you can tweak

    grid_vals = np.arange(0.0, max_total_dop + step, step)

    rows = []
    for Sc_val in grid_vals:
        for Zr_val in grid_vals:
            for Cr_val in grid_vals:
                total_dop = Sc_val + Zr_val + Cr_val
                if total_dop > max_total_dop:
                    continue

                tm_sum = sum(BASE_TM.values())  # 8.0
                Al_val = 100.0 - tm_sum - total_dop
                if Al_val < 80.0 or Al_val > 96.0:
                    # optional sanity bounds
                    continue

                comp = {
                    "Scenario": "case1_TM_fixed_Al_var",
                    "Al": Al_val,
                    "Ti": BASE_TM["Ti"],
                    "Fe": BASE_TM["Fe"],
                    "Co": BASE_TM["Co"],
                    "Ni": BASE_TM["Ni"],
                    "Sc": Sc_val,
                    "Zr": Zr_val,
                    "Cr": Cr_val,
                    "Process": PROCESS_CODE,
                }
                rows.append(comp)

    df = pd.DataFrame(rows)
    print("Case 1 designspace size:", df.shape)
    return df


# ---------------------------
# Case 2: Add Sc/Zr/Cr at expense of TM, Al fixed at 92 at.%
# ---------------------------
def build_case2_designspace():
    """
    Base: Al=92, Ti=2,Fe=2,Co=2,Ni=2 (TM_sum=8).
    Add Sc,Zr,Cr with total_dop <= max_dop.
    TM_sum_new = 8 - total_dop, scaled proportionally from base.
    """
    BASE_TM = {"Ti": 2.0, "Fe": 2.0, "Co": 2.0, "Ni": 2.0}
    TM_base_sum = sum(BASE_TM.values())  # 8.0
    dopants = ["Sc", "Zr", "Cr"]
    step = 0.5
    max_total_dop = 4.0

    grid_vals = np.arange(0.0, max_total_dop + step, step)

    rows = []
    for Sc_val in grid_vals:
        for Zr_val in grid_vals:
            for Cr_val in grid_vals:
                total_dop = Sc_val + Zr_val + Cr_val
                if total_dop > max_total_dop:
                    continue

                TM_sum_new = TM_base_sum - total_dop
                if TM_sum_new < 0.0:
                    continue

                # scale TM to preserve 2:2:2:2 but reduced sum
                scale = TM_sum_new / TM_base_sum if TM_base_sum > 0 else 0.0
                Ti_val = BASE_TM["Ti"] * scale
                Fe_val = BASE_TM["Fe"] * scale
                Co_val = BASE_TM["Co"] * scale
                Ni_val = BASE_TM["Ni"] * scale

                Al_val = 92.0  # fixed
                total = Al_val + Ti_val + Fe_val + Co_val + Ni_val + total_dop
                # small numerical noise tolerance
                if abs(total - 100.0) > 1e-6:
                    continue

                comp = {
                    "Scenario": "case2_Al_fixed_TM_var",
                    "Al": Al_val,
                    "Ti": Ti_val,
                    "Fe": Fe_val,
                    "Co": Co_val,
                    "Ni": Ni_val,
                    "Sc": Sc_val,
                    "Zr": Zr_val,
                    "Cr": Cr_val,
                    "Process": PROCESS_CODE,
                }
                rows.append(comp)

    df = pd.DataFrame(rows)
    print("Case 2 designspace size:", df.shape)
    return df


# ---------------------------
# Case 3: Full enumeration of TM + Al + Sc/Zr/Cr
# ---------------------------
def build_case3_designspace():
    """
    Ti,Fe,Co,Ni in {0,1,2,3,4}.
    Sc,Zr,Cr in {0,1,2} with dop_total <= 2.
    Al = 100 - TM_sum - dop_total.
    Filter: 80 <= Al <= 96, TM_sum >= 4 (so not trivial).
    """
    TM_values = [0, 1, 2, 3, 4]
    dop_values = [0, 1, 2]

    rows = []
    for Ti_val in TM_values:
        for Fe_val in TM_values:
            for Co_val in TM_values:
                for Ni_val in TM_values:
                    TM_sum = Ti_val + Fe_val + Co_val + Ni_val
                    if TM_sum < 4 or TM_sum > 20:
                        continue

                    for Sc_val in dop_values:
                        for Zr_val in dop_values:
                            for Cr_val in dop_values:
                                dop_total = Sc_val + Zr_val + Cr_val
                                if dop_total > 2:
                                    continue

                                Al_val = 100.0 - TM_sum - dop_total
                                if Al_val < 80.0 or Al_val > 96.0:
                                    continue

                                comp = {
                                    "Scenario": "case3_full_enumeration",
                                    "Al": Al_val,
                                    "Ti": float(Ti_val),
                                    "Fe": float(Fe_val),
                                    "Co": float(Co_val),
                                    "Ni": float(Ni_val),
                                    "Sc": float(Sc_val),
                                    "Zr": float(Zr_val),
                                    "Cr": float(Cr_val),
                                    "Process": PROCESS_CODE,
                                }
                                rows.append(comp)

    df = pd.DataFrame(rows)
    print("Case 3 designspace size:", df.shape)
    return df


# ---------------------------
# Common prediction helper
# ---------------------------
def predict_for_case(df_case: pd.DataFrame, case_label: str):
    """
    Given a designspace df with composition + Process,
    add any missing feature columns, run the three models,
    compute Q, and save full + top10 CSVs.
    """
    # make a copy so we can safely modify
    df = df_case.copy()

    # ensure all feature columns exist
    for col in all_features:
        if col not in df.columns:
            df[col] = 0.0

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

    # Q
    df["Q"] = compute_Q(df["YS"], df["UTS"], df["Elongation"])

    # Save all
    all_out = f"{case_label}_all_predictions.csv"
    df.to_csv(all_out, index=False)
    print(f"Saved full predictions for {case_label} -> {all_out} (N={len(df)})")

    # Save top 10 by Q
    top10 = df.sort_values("Q", ascending=False).head(10)
    top_out = f"{case_label}_top10_by_Q.csv"
    top10.to_csv(top_out, index=False)
    print(f"Saved top-10 by Q for {case_label} -> {top_out}")

    return df, top10


# ---------------------------
# Run all three cases
# ---------------------------
if __name__ == "__main__":
    # Case 1
    case1_df = build_case1_designspace()
    predict_for_case(case1_df, "case1")

    # Case 2
    case2_df = build_case2_designspace()
    predict_for_case(case2_df, "case2")

    # Case 3
    case3_df = build_case3_designspace()
    predict_for_case(case3_df, "case3")
