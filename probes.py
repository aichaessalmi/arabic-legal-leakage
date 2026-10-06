# -*- coding: utf-8 -*-
"""
Leakage probes (Table "Leakage probes" of the paper): classifiers restricted to one
source of information, evaluated with the same 5 random splits.

    python probes.py --masked DATASET_MASKED        # keyword-masked corpus (v1)
    python probes.py --masked DATASET_MASKED_V2     # structurally masked corpus (v2)
"""
import argparse, os, re, glob
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

FILE_RE = re.compile(r"(\d{4})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d+)")
MONTH_YEAR_RE = re.compile(r"بتاريخ\s*:?\s*\d{1,2}\s*/?\s*\S+\s*/?\s*((?:19|20)\d\d)")


def load(root):
    rows = []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if os.path.isdir(d):
            for f in sorted(glob.glob(os.path.join(d, "*_cleaned.txt"))):
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    rows.append((f"{cls}/{os.path.basename(f)}", cls, fh.read()))
    return pd.DataFrame(rows, columns=["id", "label", "text"]).sort_values("id").reset_index(drop=True)


def year(t):
    m = MONTH_YEAR_RE.search(t[:600])
    if m:
        return int(m.group(1))
    m = re.search(r"\b(20[0-2]\d)\b", t[:300])
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 7, 2024, 123])
    a = ap.parse_args()
    df = load(a.masked); y = df.label.values
    def chamber_digit(t):
        m = FILE_RE.search(t[:800])
        return m.group(3) if m else "none"
    code = df.text.map(chamber_digit).values
    yr = df.text.map(year).values.reshape(-1, 1)
    views = {"Header line only": df.text.map(lambda t: t.split("\n")[0][:300]).values,
             "Last 400 characters only": df.text.map(lambda t: t[-400:]).values,
             "Full text": df.text.values}
    res = {}
    for seed in a.seeds:
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=seed)
        maj = pd.Series(y[tr]).mode()[0]
        out = {"Majority class": np.full(len(te), maj)}
        mp = pd.crosstab(code[tr], y[tr]).idxmax(axis=1).to_dict()
        out["File-number chamber digit only (rule)"] = np.array([mp.get(c, maj) for c in code[te]])
        out["Decision year only (tree)"] = DecisionTreeClassifier(max_depth=6, random_state=0).fit(yr[tr], y[tr]).predict(yr[te])
        for name, X in views.items():
            v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=30000, min_df=2)
            out[name] = LinearSVC().fit(v.fit_transform(X[tr]), y[tr]).predict(v.transform(X[te]))
        for k, p in out.items():
            res.setdefault(k, []).append((accuracy_score(y[te], p) * 100, f1_score(y[te], p, average="macro")))
    print(f"Corpus: {a.masked}")
    for k, v in res.items():
        v = np.array(v)
        print(f"{k:<40} acc = {v[:,0].mean():6.2f} ± {v[:,0].std(ddof=1):.2f}   macro-F1 = {v[:,1].mean():.3f} ± {v[:,1].std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
