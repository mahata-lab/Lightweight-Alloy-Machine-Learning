import numpy as np
import pandas as pd
import joblib

# ============================================================
# Paths / constants
# ============================================================
MODEL_PATH = "models_cfg10b_highEI_softWeights_v4.joblib"
PROCESS_CODE = 1000000  # same process code as training (e.g., LPBF-like)

# ============================================================
# Load model dict and unpack per-target models
# ============================================================
print("Loading model dict...")
model_dict = joblib.load(MODEL_PATH)
print("Top-level keys:", model_dict.keys())  # expected: ['YS','UTS','Elongation']


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

# Union of all feature names used by any target
all_features = sorted(set(ys_features) | set(uts_features) | set(el_features))
print("Total unique features across all targets:", len(all_features))


# ============================================================
# Q definition  (EDIT THIS if your final paper uses a different formula)
# ============================================================
def compute_Q(ys, uts, elong):
    """
    Strength–ductility synergy metric.
    Replace with your exact definition from the paper if different.
    """
    return uts + 30.0 * elong


# ============================================================
# CASE 1: Add Sc/Zr/Cr at expense of Al, TM fixed at 8 at.%
# ============================================================
def build_case1_designspace():
    """
    Base: Ti = 2, Fe = 2, Co = 2, Ni = 2 (TM_sum=8).
    Add Sc, Zr, Cr s.t. total_dop <= 4 at.%
    Al = 100 - 8 - total_dop.
    Optional sanity bounds on Al are applied (80–96).
    """
    BASE_TM = {"Ti": 2.0, "Fe": 2.0, "Co": 2.0, "Ni": 2.0}
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

                tm_sum = sum(BASE_TM.values())  # 8.0
                Al_val = 100.0 - tm_sum - total_dop
                if Al_val < 80.0 or Al_val > 96.0:
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


# ============================================================
# CASE 2: Add Sc/Zr/Cr at expense of TM, Al fixed at 92 at.%
# ============================================================
def build_case2_designspace():
    """
    Base: Al=92, Ti=2,Fe=2,Co=2,Ni=2 (TM_sum=8).
    Add Sc,Zr,Cr with total_dop <= 4 at.%.
    TM_sum_new = 8 - total_dop, scaled proportionally from base (2:2:2:2).
    Al fixed at 92.
    """
    BASE_TM = {"Ti": 2.0, "Fe": 2.0, "Co": 2.0, "Ni": 2.0}
    TM_base_sum = sum(BASE_TM.values())  # 8.0

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

                # scale each TM to preserve 2:2:2:2 but reduced sum
                scale = TM_sum_new / TM_base_sum if TM_base_sum > 0 else 0.0
                Ti_val = BASE_TM["Ti"] * scale
                Fe_val = BASE_TM["Fe"] * scale
                Co_val = BASE_TM["Co"] * scale
                Ni_val = BASE_TM["Ni"] * scale

                Al_val = 92.0  # fixed
                total = Al_val + Ti_val + Fe_val + Co_val + Ni_val + total_dop
                if abs(total - 100.0) > 1e-6:
                    # numerical sanity check
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


# ============================================================
# CASE 3: Full enumeration – EXTENDED to Al = 80–99%, dopants ≤ 5%
# ============================================================
def build_case3_designspace():
    """
    EXTENDED Case 3 (final version):
    - Al allowed from 80–99 at.% (computed from closure)
    - Ti,Fe,Co,Ni in {0,1,2,3,4,5,6}
    - Sc,Zr,Cr in {0,1,2,3,4,5}
    - Total dopants (Sc+Zr+Cr) <= 5
    - Total composition = 100
    - TM_sum (Ti+Fe+Co+Ni) <= 24
    - Al must remain between 80–99 after closure
    - Avoid trivial alloys by requiring TM_sum >= 2
    """

    TM_vals = [0, 1, 2, 3, 4, 5, 6]
    DOP_vals = [0, 1, 2, 3, 4, 5]

    rows = []

    for Ti_val in TM_vals:
        for Fe_val in TM_vals:
            for Co_val in TM_vals:
                for Ni_val in TM_vals:

                    TM_sum = Ti_val + Fe_val + Co_val + Ni_val
                    if TM_sum < 2 or TM_sum > 24:
                        continue

                    for Sc_val in DOP_vals:
                        for Zr_val in DOP_vals:
                            for Cr_val in DOP_vals:

                                dop_total = Sc_val + Zr_val + Cr_val
                                if dop_total > 5:
                                    continue

                                # closure
                                Al_val = 100 - TM_sum - dop_total
                                if Al_val < 80 or Al_val > 99:
                                    continue

                                comp = {
                                    "Scenario": "case3_full_ext_Al80_99",
                                    "Al": float(Al_val),
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
    print("Extended Case 3 (Al 80–99%) designspace size:", df.shape)
    return df


# ============================================================
# Common prediction helper
# ============================================================
def predict_for_case(df_case: pd.DataFrame, case_label: str):
    """
    Given a designspace df with composition + Process,
    add any missing feature columns, run the three models,
    compute Q, and save full + top10 CSVs.

    Returns (df_with_predictions, df_top10).
    """
    df = df_case.copy()

    # ensure all feature columns exist
    for col in all_features:
        if col not in df.columns:
            df[col] = 0.0

    # Predict YS
    X_ys = df[ys_features].copy()
    X_ys_imp = ys_imputer.transform(X_ys)
    df["YS"] = ys_model.predict(X_ys_imp)

    # Predict UTS
    X_uts = df[uts_features].copy()
    X_uts_imp = uts_imputer.transform(X_uts)
    df["UTS"] = uts_model.predict(X_uts_imp)

    # Predict Elongation
    X_el = df[el_features].copy()
    X_el_imp = el_imputer.transform(X_el)
    df["Elongation"] = el_model.predict(X_el_imp)

    # Compute Q
    df["Q"] = compute_Q(df["YS"], df["UTS"], df["Elongation"])

    # Save full predictions
    all_out = f"{case_label}_all_predictions.csv"
    df.to_csv(all_out, index=False)
    print(f"Saved full predictions for {case_label} -> {all_out} (N={len(df)})")

    # Save top10 by Q
    top10 = df.sort_values("Q", ascending=False).head(10)
    top_out = f"{case_label}_top10_by_Q.csv"
    top10.to_csv(top_out, index=False)
    print(f"Saved top-10 by Q for {case_label} -> {top_out}")

    return df, top10


# ============================================================
# Main: build 3 cases and predict
# ============================================================
if __name__ == "__main__":
    # Case 1
    case1_df = build_case1_designspace()
    case1_all, case1_top = predict_for_case(case1_df, "case1")

    # Case 2
    case2_df = build_case2_designspace()
    case2_all, case2_top = predict_for_case(case2_df, "case2")

    # Case 3 (extended dopants, Al 80–99%)
    case3_df = build_case3_designspace()
    case3_all, case3_top = predict_for_case(case3_df, "case3")

    # Quick sanity print
    print("\nSummary of UTS / Elongation ranges:")
    for label, df in [
        ("Case1", case1_all),
        ("Case2", case2_all),
        ("Case3_ext_Al80_99", case3_all),
    ]:
        uts_min, uts_max = df["UTS"].min(), df["UTS"].max()
        e_min, e_max = df["Elongation"].min(), df["Elongation"].max()
        print(
            f"{label}: UTS [{uts_min:.1f}, {uts_max:.1f}] MPa, "
            f"E [{e_min:.2f}, {e_max:.2f}] %"
        )
