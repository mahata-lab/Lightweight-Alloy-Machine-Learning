import joblib

model_dict = joblib.load("models_cfg10b_highEI_softWeights_v4.joblib")
print("Top-level keys:", model_dict.keys())

for target in ["YS", "UTS", "Elongation"]:
    obj = model_dict[target]
    print(f"\n=== {target} entry ===")
    print("Type:", type(obj))
    if isinstance(obj, dict):
        print("Inner keys:", obj.keys())
    else:
        print("Not a dict, has attributes:", dir(obj)[:20])

