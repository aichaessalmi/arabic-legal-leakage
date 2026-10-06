"""
Agrège les métriques de tous les seeds : moyenne ± écart-type par modèle et
condition, plus le delta (masqué - original) apparié par seed.

    python aggregate.py --out results
Produit : results/summary.csv et results/summary.md (tableau prêt pour l'article)
"""
import argparse
import glob
import json
import os

import pandas as pd
from scipy import stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    a = p.parse_args()

    rows = [json.load(open(f)) for f in glob.glob(os.path.join(a.out, "metrics", "*.json"))]
    df = pd.DataFrame(rows)
    metrics = ["accuracy", "f1_macro", "f1_weighted"]

    lines = ["| Modèle | n seeds | Acc. orig. | Acc. masq. | F1-mac. orig. | F1-mac. masq. "
             "| ΔF1-mac. (pp) | p (t apparié) |",
             "|---|---|---|---|---|---|---|---|"]
    summary = []
    for model, g in df.groupby("model"):
        piv = g.pivot_table(index="seed", columns="condition", values=metrics)
        rec = {"model": model, "n_seeds": len(piv)}
        for m in metrics:
            for c in ("original", "masked"):
                if (m, c) in piv:
                    rec[f"{m}_{c}_mean"] = piv[(m, c)].mean()
                    rec[f"{m}_{c}_std"] = piv[(m, c)].std(ddof=1) if len(piv) > 1 else 0.0
        pval, dmean, dstd = float("nan"), float("nan"), float("nan")
        if ("f1_macro", "original") in piv and ("f1_macro", "masked") in piv:
            paired = piv[[("f1_macro", "original"), ("f1_macro", "masked")]].dropna()
            d = (paired[("f1_macro", "masked")] - paired[("f1_macro", "original")]) * 100
            dmean, dstd = d.mean(), (d.std(ddof=1) if len(d) > 1 else 0.0)
            if len(d) > 1:
                pval = stats.ttest_rel(paired[("f1_macro", "masked")],
                                       paired[("f1_macro", "original")]).pvalue
        rec.update({"delta_f1_macro_pp_mean": dmean, "delta_f1_macro_pp_std": dstd, "p_value": pval})
        summary.append(rec)

        def fmt(m, c, scale=1.0, nd=4):
            k = f"{m}_{c}"
            if f"{k}_mean" not in rec:
                return "–"
            return f"{rec[k + '_mean'] * scale:.{nd}f} ± {rec[k + '_std'] * scale:.{nd}f}"

        lines.append(f"| {model} | {len(piv)} | {fmt('accuracy', 'original', 100, 2)} "
                     f"| {fmt('accuracy', 'masked', 100, 2)} | {fmt('f1_macro', 'original')} "
                     f"| {fmt('f1_macro', 'masked')} | {dmean:+.2f} ± {dstd:.2f} | {pval:.3f} |")

    pd.DataFrame(summary).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
