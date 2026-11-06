"""Impute missing failure_reasons for failed repairs from failure_reason_open.

failure_reason_open is filled in almost only when the dropdown was left blank
(13 rows have both), so it is labeled by hand in data/open_text_labels.csv and
a classifier is trained on multilingual sentence embeddings of that sample.
An extra class, "No reason stated", catches entries that only describe the
fault ("motor defect"); those rows stay unclassified.

defect_found / problem_description were tried as well and dropped: they say
what broke, not why the repair failed (30% held-out accuracy vs a 21%
majority baseline, macro-F1 0.23).
Usage: python classify_failures.py [data_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
CACHE = Path("cache")
CACHE.mkdir(exist_ok=True)
MODEL = "intfloat/multilingual-e5-small"
NO_REASON = "No reason stated"
MIN_CONF = 0.4  # ~79% cross-validated accuracy, keeps about two thirds

# --- Load (same cleaning as analysis.py) --------------------------------------
repairs = pd.read_csv(DATA / "repairs.csv", parse_dates=["repair_date"])
text = pd.read_csv(DATA / "repairs_text.csv", low_memory=False)
assert repairs["repair_id"].equals(text["repair_id"]), "files are not row-aligned"
df = pd.concat([repairs, text.drop(columns="repair_id")], axis=1)
df["row_key"] = df["repair_id"] + "#" + df.index.astype(str)
df["repaired"] = df["repaired"].replace({"ja": "yes"})
df = df[(df["repaired"] == "no") & (df["repair_date"] <= "2026-04-07")].copy()

o = df["failure_reason_open"].fillna("").astype(str).str.strip()
df["text_open"] = o.mask(o.str.lower().isin(["na", "n/a", "-", "?"]), "")
labeled = df["failure_reasons"].notna()
target = ~labeled & df["text_open"].ne("")
print(f"{len(df):,} failed repairs; {(~labeled).sum():,} without a dropdown reason, "
      f"{target.sum():,} of those have open text")

# --- Embeddings (cached by unique text) ---------------------------------------
model = SentenceTransformer(MODEL, device="cpu")


def embed(texts, name):
    path = CACHE / f"emb_{name}.npz"
    uniq = pd.Index(pd.unique(texts[texts != ""]))
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if list(z["texts"]) == list(uniq):
            return uniq, z["emb"]
    print(f"embedding {len(uniq):,} unique {name} texts...")
    emb = model.encode(["query: " + t for t in uniq], batch_size=128,
                       normalize_embeddings=True, show_progress_bar=True)
    np.savez(path, texts=np.array(uniq, dtype=object), emb=emb)
    return uniq, emb


uniq, emb = embed(df["text_open"], "open")

# --- Train on the hand-labeled sample ------------------------------------------
gold = pd.read_csv(DATA / "open_text_labels.csv")
X = emb[uniq.get_indexer(gold["text_open"])]
y = gold["label"].values
print(f"\n{len(gold)} hand-labeled texts")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)  # smallest class has 3
majority = pd.Series(y).mode()[0]
print(f"Baseline (always '{majority}'): acc {np.mean(y == majority):.3f}")
scores = {}
for C in [10, 30, 100, 300]:
    pr = cross_val_predict(LogisticRegression(C=C, max_iter=5000, class_weight="balanced"), X, y, cv=cv)
    scores[C] = f1_score(y, pr, average="macro")
    print(f"C={C:<3} acc {np.mean(pr == y):.3f}  macro-F1 {scores[C]:.3f}")
best_C = max(scores, key=scores.get)
clf = LogisticRegression(C=best_C, max_iter=5000, class_weight="balanced")
proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
classes = np.unique(y)
pr = classes[proba.argmax(1)]
print(f"\nCross-validated report, C={best_C}\n",
      classification_report(y, pr, digits=3, zero_division=0))
conf = proba.max(1)
for t in [0.3, 0.4, 0.5]:
    k = conf >= t
    print(f"confidence >= {t}: keeps {k.mean():.0%}, acc {np.mean(pr[k] == y[k]):.3f}")
clf.fit(X, y)

# --- Impute ------------------------------------------------------------------
P = clf.predict_proba(emb[uniq.get_indexer(df.loc[target, "text_open"])])
df["reason"] = df["failure_reasons"]
df["source"] = np.where(labeled, "dropdown", "none")
df["confidence"] = np.where(labeled, 1.0, np.nan)
pred = clf.classes_[P.argmax(1)]
stated = (pred != NO_REASON) & (P.max(1) >= MIN_CONF)
idx = df.index[target][stated]
df.loc[idx, "reason"] = pred[stated]
df.loc[idx, "confidence"] = P.max(1)[stated]
df.loc[idx, "source"] = "open_text"
print(f"\nOpen-text rows left unclassified ('{NO_REASON}' or confidence < {MIN_CONF}): "
      f"{(~stated).sum():,} of {len(pred):,}")

print("\nSource of reason for failed repairs\n", df["source"].value_counts().to_string())
mix = pd.DataFrame({
    "dropdown": df.loc[df["source"] == "dropdown", "reason"].value_counts(normalize=True),
    "open_text": df.loc[df["source"] == "open_text", "reason"].value_counts(normalize=True),
    "combined": df["reason"].value_counts(normalize=True)})
print("\nReason mix by source\n", mix.round(3).fillna(0).to_string())

out = df[["row_key", "failure_reasons", "reason", "source", "confidence", "text_open"]]
out.to_csv(DATA / "failure_reasons_imputed.csv", index=False)
print(f"\nWrote {DATA / 'failure_reasons_imputed.csv'}")
