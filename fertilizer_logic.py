from pandas import read_csv
import pickle
import numpy as np


def detect_deficiency(deficiency):
    column = ["N", "P", "K", "OC", "B", "Cu", "Fe", "Mn", "S", "Zn"]
    for i in range(len(column)):
        deficiency.append(df.loc[df["State/UT"] == State, column[i]].iloc[0])

    return deficiency

def fertilizer_recommendation(deficiency):
    if sample is None:
        return generalized_fertilizers(deficiency), generalized_micronutrients(deficiency)
    else:
        return ml_fertilizer(sample), generalized_micronutrients(deficiency)


def ml_fertilizer(sample):
    with open("models/fertilizer_model.pkl", "rb") as f:
        model = pickle.load(f)

    proba = model.predict_proba(sample)[0]
    classes = model.classes_
    top_indices = np.argsort(proba)[::-1][:3]
    fertilizers = {}
    for i in top_indices:
        key = classes[i]
        value = f"{round(proba[i] * 100, 2)}%"
        fertilizers[key] = value

    return fertilizers


def generalized_fertilizers(deficiency):

    N,P,K,OC = deficiency[0], deficiency[1],deficiency[2],deficiency[3]

    LOW = {"very low", "low"}
    OK = {"medium", "high", "very high"}

    low_macro = {
        "N" : N in LOW,
        "P": P in LOW,
        "K": K in LOW
    }

    low_count = sum(low_macro.values())

    fertilizer = []

    if low_count == 0:
        fertilizer.append("no fertilizer needed")

    if OC in LOW:
        if low_count == 2:
            fertilizer.append("organic matter + npk_complex")
        else:
            fertilizer.append("organic matter")

    elif low_count == 3:
        fertilizer.append("npk_complex")

    elif low_macro["N"] and low_macro["P"]:
        fertilizer.append("dap")

    elif low_macro["N"] and low_macro["K"]:
        fertilizer.append("npk_complex")

    elif low_macro["N"]:
        fertilizer.append("urea")

    elif low_macro["P"]:
        fertilizer.append("ssp")

    elif low_macro["K"]:
        fertilizer.append("mop")

    else:
        fertilizer.append("No recommendation")

    print(fertilizer)


def generalized_micronutrients(deficiency):
    micro_map = {
        "4": "zinc_sulphate",
        "5": "borax",
        "6": "ferrous_sulphate",
        "7": "manganese_sulphate",
        "8": "copper_sulphate",
        "9": "sulphur_bentonite"
    }

    micronutrients = []

    for i in range(4, len(deficiency), 1):
        if deficiency[i] == "deficient":
            micronutrients.append(micro_map[str(i)])

    if micronutrients:
        print(micronutrients)
    else:
        micronutrients.append("no_micronutrient_needed")
        print(micronutrients)

if __name__ == "__main__":
    df = read_csv("Dataset/state_soil_summary.csv")
    State = "Delhi"
    sample = None
    Deficiency = []
    detect_deficiency(Deficiency)
    fertilizer_recommendation(Deficiency)


