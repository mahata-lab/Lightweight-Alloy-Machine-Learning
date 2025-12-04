import pandas as pd

# Base Al92Ti2Fe2Co2Ni2 composition (at.%)
BASE_Al = 92.0
BASE_Ti = 2.0
BASE_Fe = 2.0
BASE_Co = 2.0
BASE_Ni = 2.0

micro_elements = ["Sc", "Cr", "Zr", "Mg", "Ag", "V", "Cu", "Mn"]
deltas = [0.0, 0.25, 0.5, 1.0, 2.0]

# Numeric Process code: use the same code as the Al92 training row
# In your training CSV, that row has Process = 1000000
PROCESS_CODE = 1000000

out_rows = []

for X in micro_elements:
    for delta in deltas:
        al_at = BASE_Al - delta
        if al_at < 0:
            continue

        row = {
            "Alloy Name": f"Al92Ti2Fe2Co2Ni2+{delta}{X}",
            "Serial Number": -1,   # placeholder
            "Process": PROCESS_CODE,
        }

        # Initialize all element columns in training set to 0.0
        element_cols = [
            "Al","Ag","B","Be","Bi","C","Ca","Cd","Ce","Co","Cr","Cu","Er","Eu",
            "Fe","Ga","Gd","Hf","In","Li","Mg","Mn","Na","Ni","Pb","Sb","Sc",
            "Si","Sn","Sr","Ti","V","Y","Yb","Zn","Zr"
        ]
        for e in element_cols:
            row[e] = 0.0

        # Set base quinary composition
        row["Al"] = al_at
        row["Ti"] = BASE_Ti
        row["Fe"] = BASE_Fe
        row["Co"] = BASE_Co
        row["Ni"] = BASE_Ni

        # Add microalloy X
        row[X] = delta

        # Add bookkeeping
        row["MicroElement"] = X
        row["Delta_atpct"]  = delta

        out_rows.append(row)

df = pd.DataFrame(out_rows)
df.to_csv("Al92_microalloying_designspace_composition_only.csv", index=False)
print("Saved Al92_microalloying_designspace_composition_only.csv with shape:", df.shape)
print(df.head())
