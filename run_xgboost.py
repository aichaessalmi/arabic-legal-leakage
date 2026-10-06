# -*- coding: utf-8 -*-
"""
XGBoost + TF-IDF on the original and masked corpora (configuration of the paper).

- TF-IDF: unigrams + bigrams, 30,000 features, sublinear tf, 50 Arabic stop words
- XGBoost: lr 0.1, max depth 6, up to 300 trees, early stopping (20 rounds) on a
  validation set = 10% of the training set (the test set is never used for selection)
- Same Arabic normalisation for every corpus
- Random stratified split (5 seeds) or per-chamber chronological split

Examples
    python run_xgboost.py --orig CORPUS_PROCESSED --masked DATASET_MASKED_V2 --out results_v2
    python run_xgboost.py --orig CORPUS_PROCESSED --masked DATASET_MASKED_V2 --out results_chrono \
        --split release/split_chronological.csv --seeds 42 13 7
Outputs: <out>/predictions/xgboost__<condition>__seed<s>.csv and <out>/metrics/*.json
"""
import argparse, os, re
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from common import load_pair, save_run

STOP = ["في","من","على","إلى","عن","أن","إن","و","ب","ل","ما","لا","التي","الذي","هذا","هذه","ذلك","تلك",
        "كان","كانت","قد","لم","لن","ثم","أو","بل","حتى","إذا","كل","بعد","قبل","عند","غير","بين","وقد",
        "فيه","فيها","منه","منها","به","بها","له","لها","هو","هي","أيضا","حيث","كما","إلا"]


def norm(t):
    t = re.sub(r'[إأآا]', 'ا', t); t = re.sub(r'ى', 'ي', t); t = re.sub(r'ؤ', 'و', t)
    t = re.sub(r'ئ', 'ي', t); t = re.sub(r'ة', 'ه', t); t = re.sub(r'[ً-ٟ]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True)
    p.add_argument("--masked", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 7, 2024, 123])
    p.add_argument("--split", default=None, help="CSV (id,split) for a fixed split, e.g. release/split_chronological.csv")
    p.add_argument("--n-jobs", type=int, default=-1)
    a = p.parse_args()

    ids, t_orig, t_mask, labels, classes = load_pair(a.orig, a.masked, do_normalize=False)
    ids = np.asarray(ids)
    texts = {"original": np.asarray([norm(t) for t in t_orig], dtype=object),
             "masked": np.asarray([norm(t) for t in t_mask], dtype=object)}
    le = LabelEncoder().fit(labels); y = le.transform(labels)
    fixed = None
    if a.split:
        s = pd.read_csv(a.split, encoding="utf-8-sig").set_index("id").loc[ids, "split"].values
        fixed = tuple(np.where(s == k)[0] for k in ("train", "val", "test"))

    for seed in a.seeds:
        if fixed:
            tr, va, te = fixed
        else:
            idx = np.arange(len(ids))
            tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
            tr, va = train_test_split(tr, test_size=0.1, random_state=seed, stratify=y[tr])
        for cond, X in texts.items():
            vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, stop_words=STOP)
            A = vec.fit_transform(X[tr])
            clf = xgb.XGBClassifier(booster="gbtree", tree_method="hist", learning_rate=0.1, max_depth=6,
                                    n_estimators=300, min_child_weight=1, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                                    colsample_bytree=0.8, subsample=0.8, objective="multi:softprob",
                                    eval_metric="mlogloss", early_stopping_rounds=20, random_state=seed,
                                    n_jobs=a.n_jobs)
            clf.fit(A, y[tr], eval_set=[(vec.transform(X[va]), y[va])], verbose=False)
            pred = clf.predict(vec.transform(X[te]))
            save_run(a.out, "xgboost", cond, seed, ids[te].tolist(),
                     le.inverse_transform(y[te]).tolist(), le.inverse_transform(pred).tolist(),
                     extra={"best_iteration": int(clf.best_iteration)})


if __name__ == "__main__":
    main()
