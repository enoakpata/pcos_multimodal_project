import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

img = pd.read_csv("outputs/meta/image_probs.csv")
clin = pd.read_csv("outputs/meta/clinical_probs.csv")

# align sizes (since datasets differ)
n = min(len(img), len(clin))

X = pd.DataFrame({
    "p_img": img["p_img"][:n],
    "p_clin": clin["p_clin"][:n]
})

y = clin["PCOS (Y/N)"][:n]

meta = LogisticRegression()
meta.fit(X, y)

joblib.dump(meta, "outputs/meta/meta_model.pkl")

print("Meta learner trained.")
