"""
Test de McNemar entre toutes les paires de modèles sur le MÊME test set (seed 42),
et entre original et masqué pour chaque modèle.

Pour Jais : exportez ses prédictions au même format que les autres :
    results/predictions/jais13b__masked__seed42.csv   (colonnes : id, y_true, y_pred)
    results/predictions/jais13b__original__seed42.csv
Les ids doivent être au format "<chambre>/<fichier>.txt".

    python mcnemar.py --out results --seed 42
"""
import argparse
import glob
import itertools
import os

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def load(out, seed):
    runs = {}
    for f in glob.glob(os.path.join(out, "predictions", f"*__seed{seed}.csv")):
        model, cond, _ = os.path.basename(f)[:-4].split("__")
        df = pd.read_csv(f)
        runs[(model, cond)] = df.set_index("id").assign(ok=lambda d: d.y_true == d.y_pred)["ok"]
    return runs


def test(a, b):
    common = a.index.intersection(b.index)
    if len(common) != len(a) or len(common) != len(b):
        print(f"  ⚠ test sets différents ({len(a)} / {len(b)} / communs {len(common)})")
    a, b = a.loc[common], b.loc[common]
    n01 = int((a & ~b).sum())   # A correct, B faux
    n10 = int((~a & b).sum())   # A faux, B correct
    table = [[int((a & b).sum()), n01], [n10, int((~a & ~b).sum())]]
    res = mcnemar(table, exact=(n01 + n10) < 25)
    return n01, n10, res.pvalue, len(common)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    runs = load(a.out, a.seed)
    rows = []

    for cond in ("original", "masked"):
        models = sorted(m for m, c in runs if c == cond)
        for m1, m2 in itertools.combinations(models, 2):
            n01, n10, pv, n = test(runs[(m1, cond)], runs[(m2, cond)])
            rows.append({"comparaison": f"{m1} vs {m2}", "condition": cond, "n": n,
                         "A_seul_correct": n01, "B_seul_correct": n10, "p_value": pv})

    for m in sorted({m for m, _ in runs}):
        if (m, "original") in runs and (m, "masked") in runs:
            n01, n10, pv, n = test(runs[(m, "original")], runs[(m, "masked")])
            rows.append({"comparaison": f"{m}: original vs masqué", "condition": "–", "n": n,
                         "A_seul_correct": n01, "B_seul_correct": n10, "p_value": pv})

    df = pd.DataFrame(rows)
    df["significatif_0.05"] = df.p_value < 0.05
    df.to_csv(os.path.join(a.out, "mcnemar.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
