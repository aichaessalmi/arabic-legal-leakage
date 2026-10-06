# -*- coding: utf-8 -*-
"""
Figure 8 (révision) : SHAP pour XGBoost + TF-IDF, original vs masqué v2.

- Même configuration XGBoost que le nouveau Tableau 5 (seed 42, early stopping
  sur 10 % du train, jamais sur le test).
- Valeurs SHAP exactes (TreeSHAP) calculées par XGBoost lui-même
  (pred_contribs=True), sur 200 documents de test stratifiés.
- Définition objective d'un token de fuite : n-gramme dont la fréquence
  documentaire chute de plus de 90 % entre le corpus original et le corpus
  masqué v2 (= ce que le masquage a retiré). Plus de liste de mots choisie à la main.

    python shap_v2.py
Sorties : results_v2/shap/
"""
import os, re, json, glob, argparse
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

STOP = ["في","من","على","إلى","عن","أن","إن","و","ب","ل","ما","لا","التي","الذي","هذا","هذه","ذلك","تلك",
        "كان","كانت","قد","لم","لن","ثم","أو","بل","حتى","إذا","كل","بعد","قبل","عند","غير","بين","وقد",
        "فيه","فيها","منه","منها","به","بها","له","لها","هو","هي","أيضا","حيث","كما","إلا"]
EN = {"الغرفة الإجتماعية": "Social", "الغرفة الإدارية": "Administrative", "الغرفة التجارية": "Commercial",
      "الغرفة الجنائية": "Criminal", "الغرفة العقارية": "Real Estate", "الغرفة المدنية": "Civil",
      "غرفة الأحوال الشخصية و الميراث": "Personal Status"}
PLACEHOLDERS = {"masked", "num", "year", "court"}


def norm(t):
    t = re.sub(r'[إأآا]', 'ا', t); t = re.sub(r'ى', 'ي', t); t = re.sub(r'ؤ', 'و', t)
    t = re.sub(r'ئ', 'ي', t); t = re.sub(r'ة', 'ه', t); t = re.sub(r'[ً-ٟ]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def load(root):
    d = {}
    for cls in sorted(os.listdir(root)):
        p = os.path.join(root, cls)
        if os.path.isdir(p):
            for f in glob.glob(os.path.join(p, "*_cleaned.txt")):
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    d[f"{cls}/{os.path.basename(f)}"] = (fh.read(), cls)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="CORPUS_PROCESSED")
    ap.add_argument("--masked", default="DATASET_MASKED_V2")
    ap.add_argument("--out", default=os.path.join("results_v2", "shap"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-shap", type=int, default=200)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--plot-only", action="store_true", help="retrace la figure depuis les CSV déjà calculés")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.plot_only:
        res = json.load(open(os.path.join(a.out, "shap_leakage_summary.json")))
        for c in ("original", "masked"):
            res[c + "_table"] = pd.read_csv(os.path.join(a.out, f"shap_top100_{c}.csv"), encoding="utf-8-sig")
        plot(res, a.top, os.path.join(a.out, "fig8_shap_v2.png"))
        return

    O, M = load(a.orig), load(a.masked)
    ids = sorted(set(O) & set(M))
    labels = [O[i][1] for i in ids]
    texts = {"original": [norm(O[i][0]) for i in ids], "masked": [norm(M[i][0]) for i in ids]}
    le = LabelEncoder().fit(labels); y = le.transform(labels)
    idx = np.arange(len(ids))
    tr, te = train_test_split(idx, test_size=0.2, random_state=a.seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.1, random_state=a.seed, stratify=y[tr])
    sh, _ = train_test_split(te, train_size=a.n_shap, random_state=0, stratify=y[te])
    print(f"{len(ids)} documents | train {len(tr)} | val {len(va)} | test {len(te)} | SHAP sur {len(sh)}")

    res = {}
    for cond in ("original", "masked"):
        X = np.asarray(texts[cond], dtype=object)
        vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, stop_words=STOP)
        A = vec.fit_transform(X[tr])
        clf = xgb.XGBClassifier(booster="gbtree", tree_method="hist", device="cpu", learning_rate=0.1, max_depth=6,
                                n_estimators=300, min_child_weight=1, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                                colsample_bytree=0.8, subsample=0.8, objective="multi:softprob",
                                eval_metric="mlogloss", early_stopping_rounds=20, random_state=a.seed, n_jobs=-1)
        clf.fit(A, y[tr], eval_set=[(vec.transform(X[va]), y[va])], verbose=False)
        acc = float((clf.predict(vec.transform(X[te])) == y[te]).mean())
        contrib = clf.get_booster().predict(xgb.DMatrix(vec.transform(X[sh])), pred_contribs=True)
        # (n, classes, features+1) -> |SHAP| moyen sur documents et classes (sans le biais)
        imp = np.abs(contrib[:, :, :-1]).mean(axis=(0, 1))
        feats = vec.get_feature_names_out()
        # fréquence documentaire de chaque feature dans les deux corpus (sur tout le corpus)
        cv = CountVectorizer(vocabulary=vec.vocabulary_, ngram_range=(1, 2), binary=True, stop_words=STOP)
        df_o = np.asarray(cv.transform(texts["original"]).sum(0)).ravel()
        df_m = np.asarray(cv.transform(texts["masked"]).sum(0)).ravel()
        kind = np.where([any(w in PLACEHOLDERS for w in f.split()) for f in feats], "placeholder",
                        np.where((df_o >= 20) & (df_m <= 0.1 * df_o), "leakage", "content"))
        order = np.argsort(-imp)
        tab = pd.DataFrame({"feature": feats[order], "mean_abs_shap": imp[order], "type": kind[order],
                            "df_original": df_o[order], "df_masked": df_m[order]})
        tab.head(100).to_csv(os.path.join(a.out, f"shap_top100_{cond}.csv"), index=False, encoding="utf-8-sig")
        r = {"test_accuracy": acc, "best_iteration": int(clf.best_iteration)}
        for k in (30, 100):
            t = tab.head(k)
            r[f"leak_in_top{k}"] = int((t.type == "leakage").sum())
            r[f"placeholder_in_top{k}"] = int((t.type == "placeholder").sum())
        r["leak_share_total_shap_%"] = float(100 * imp[kind == "leakage"].sum() / imp.sum())
        r["placeholder_share_total_shap_%"] = float(100 * imp[kind == "placeholder"].sum() / imp.sum())
        res[cond] = r; res[cond + "_table"] = tab
        print(f"[{cond}] acc={acc:.4f} | fuite top-30={r['leak_in_top30']} top-100={r['leak_in_top100']} | "
              f"part de l'importance SHAP totale sur la fuite = {r['leak_share_total_shap_%']:.1f} % | "
              f"sur les placeholders = {r['placeholder_share_total_shap_%']:.1f} %")

    json.dump({k: v for k, v in res.items() if not k.endswith("_table")},
              open(os.path.join(a.out, "shap_leakage_summary.json"), "w"), indent=2)
    plot(res, a.top, os.path.join(a.out, "fig8_shap_v2.png"))


GLOSS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shap_glosses.csv")   # optionnel : colonnes feature,gloss


def plot(res, top, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        show = lambda t: get_display(arabic_reshaper.reshape(t))
    except ImportError:
        show = lambda t: t
    gloss = {}
    if os.path.exists(GLOSS_FILE):
        g = pd.read_csv(GLOSS_FILE, encoding="utf-8-sig"); gloss = dict(zip(g.feature, g.gloss))
    col = {"leakage": "#d62728", "placeholder": "#7f7f7f", "content": "#1f77b4"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.36 * top + 2.2))
    for ax, cond, title in ((axes[0], "original", "(a) Original corpus"), (axes[1], "masked", "(b) Fully masked corpus (v2)")):
        t = res[cond + "_table"].head(top).iloc[::-1]
        def label(f):
            parts = [("[" + w.upper() + "]") if w in PLACEHOLDERS else w for w in f.split()]
            txt = " ".join(show(p) if not p.startswith("[") else p for p in parts[::-1]) \
                if any(p.startswith("[") for p in parts) and len(parts) > 1 else \
                (parts[0] if parts[0].startswith("[") else show(f))
            return txt + (f"  ({gloss[f]})" if f in gloss else "")
        lab = [label(f) for f in t.feature]
        ax.barh(range(len(t)), t.mean_abs_shap, color=[col[k] for k in t.type])
        ax.set_yticks(range(len(t))); ax.set_yticklabels(lab, fontsize=10)
        r = res[cond]
        ax.set_title(f"{title}\nleakage features: {int((t.type == 'leakage').sum())}/{top} "
                     f"| {r['leak_share_total_shap_%']:.1f}% of total |SHAP|", fontsize=11)
        ax.set_xlabel("Mean |SHAP value| (200 test documents, all classes)")
    fig.legend(handles=[Patch(color=col["leakage"], label="Label-leaking feature (removed by masking)"),
                        Patch(color=col["placeholder"], label="Mask placeholder"),
                        Patch(color=col["content"], label="Legal content feature")],
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=300); fig.savefig(path.rsplit(".", 1)[0] + ".pdf")
    print("Figure :", path)


if __name__ == "__main__":
    main()
