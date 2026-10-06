"""
1) Recalcule l'analyse d'attention [CLS] (Algorithme 2, branche 2) avec une
   détection des tokens de fuite par positions de caractères (offsets), plus
   fiable que la comparaison de sous-mots.
2) Propose des documents candidats pour la nouvelle Figure 10 :
   correctement classés PAR LES DEUX modèles (original et masqué).
3) Trace la figure pour le document choisi.

Prérequis : avoir lancé run_encoder.py avec --keep-seed 42.

    python attention_fig10.py --orig data/original --masked data/masked \
        --models results/models --name arabertv2 --keywords leakage_keywords.txt
    python attention_fig10.py ... --plot-id "Social/12345.txt"
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import MASK_WORD, load_pair, normalize, split_indices


def spans(text, patterns):
    out = []
    for p in patterns:
        out += [(m.start(), m.end()) for m in p.finditer(text)]
    return out


def token_flags(offsets, sp):
    flags = []
    for s, e in offsets.tolist():
        flags.append(e > s and any(s < pe and e > ps for ps, pe in sp))
    return np.array(flags)


@torch.no_grad()
def analyse(model, tok, text, leak_pats, mask_pat, max_len, k, device):
    enc = tok(text, truncation=True, max_length=max_len, return_offsets_mapping=True,
              return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0]
    enc = {kk: v.to(device) for kk, v in enc.items()}
    out = model(**enc, output_attentions=True)
    att = out.attentions[-1][0].mean(0)[0].float().cpu().numpy()  # ligne [CLS], moy. des têtes
    att = att / att.sum()
    leak = token_flags(offsets, spans(text, leak_pats))
    msk = token_flags(offsets, spans(text, [mask_pat]))
    top = np.argsort(-att)[:k]
    return {
        "pred": int(out.logits.argmax(-1)),
        "leak_in_topk": int(leak[top].sum()),
        "leak_mass": float(att[leak].sum()),
        "mask_mass": float(att[msk].sum()),
        "tokens": tok.convert_ids_to_tokens(enc["input_ids"][0].cpu()),
        "att": att, "leak": leak, "msk": msk,
    }


EN = {"الغرفة الإجتماعية": "Social", "الغرفة الاجتماعية": "Social", "الغرفة الإدارية": "Administrative",
      "الغرفة التجارية": "Commercial", "الغرفة الجنائية": "Criminal", "الغرفة العقارية": "Real Estate",
      "الغرفة المدنية": "Civil", "غرفة الأحوال الشخصية و الميراث": "Personal Status"}


# Traductions anglaises des mots (forme normalisée : ة->ه, ى->ي). Ajoutez-en si besoin.
GLOSS = {"التجاريه": "commercial", "التجاري": "commercial", "الغرفه": "chamber", "التجاره": "commerce",
         "المقاوله": "enterprise", "الملف": "file", "في": "in", "عدد": "no.", "القرار": "decision",
         "مدونه": "code", "صعوبه": "difficulty", "الصعوبه": "the difficulty", "المؤرخ": "dated",
         "قضائيه": "judicial", "نعم": "yes", "يدخل": "falls within", "دين": "debt",
         "محكمه": "court", "النقض": "cassation", "الطلب": "petition", "الحكم": "judgment",
         "الطاعن": "appellant", "المطلوب": "respondent", "الاستئناف": "appeal", "الشغل": "labor",
         "التعويض": "compensation", "العقد": "contract", "الكراء": "lease", "الطلاق": "divorce",
         "النفقه": "alimony", "الارث": "inheritance", "الجريمه": "crime", "العقوبه": "penalty",
         "التحفيظ": "land registration", "العقار": "property", "الاداره": "administration", "NUM": "number"}


def merge_words(r):
    """Regroupe les sous-mots (##) en mots entiers : attention sommée, drapeaux combinés."""
    words = []
    for i, t in enumerate(r["tokens"]):
        if t in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if t.startswith("##") and words:
            w = words[-1]
            w["text"] += t[2:]; w["att"] += float(r["att"][i])
            w["leak"] |= bool(r["leak"][i]); w["msk"] |= bool(r["msk"][i])
        else:
            words.append({"text": t, "att": float(r["att"][i]),
                          "leak": bool(r["leak"][i]), "msk": bool(r["msk"][i])})
    for w in words:
        if w["msk"] or w["text"] == "MASKED":
            w["text"], w["msk"], w["leak"] = "[MASKED]", True, False
        elif w["text"] in ("NUM", "YEAR", "COURT"):
            w["text"], w["msk"], w["leak"] = f"[{w['text']}]", True, False
    return words


def plot(res_o, res_m, doc_id, true_label, id2label, path, n=15):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        show = lambda t: get_display(arabic_reshaper.reshape(t))
    except ImportError:
        show = lambda t: t
    en = lambda c: EN.get(c, c)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, r, title in ((axes[0], res_o, "(a) Model trained on original corpus"),
                         (axes[1], res_m, "(b) Model trained on masked corpus")):
        ws = sorted(merge_words(r), key=lambda w: -w["att"])[:n][::-1]
        colors = ["#d62728" if w["leak"] else ("#7f7f7f" if w["msk"] else "#1f77b4") for w in ws]
        ax.barh(range(len(ws)), [w["att"] for w in ws], color=colors)
        ax.set_yticks(range(len(ws)))
        ax.set_yticklabels([show(w["text"]) + (f"  ({GLOSS[w['text']]})" if w["text"] in GLOSS else "") for w in ws], fontsize=11)
        ax.set_xlabel("Attention weight ([CLS] token, last layer, mean over heads)")
        ax.set_title(f"{title}\ntrue: {en(true_label)} | predicted: {en(id2label[r['pred']])}", fontsize=11)
    fig.legend(handles=[Patch(color="#d62728", label="Label-leaking token"),
                        Patch(color="#7f7f7f", label="Mask placeholder ([MASKED], [NUM], ...)"),
                        Patch(color="#1f77b4", label="Legal content token")],
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=300)
    fig.savefig(path.rsplit(".", 1)[0] + ".pdf")
    print("Figure sauvegardée :", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True)
    p.add_argument("--masked", required=True)
    p.add_argument("--models", default="results/models")
    p.add_argument("--name", default="arabertv2")
    p.add_argument("--keywords", required=True, help="un terme de fuite par ligne")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-docs", type=int, default=0, help="0 = tout le test set")
    p.add_argument("--out", default="results")
    p.add_argument("--plot-id", default=None)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kws = [normalize(l.strip()) for l in open(a.keywords, encoding="utf-8") if l.strip()]
    leak_pats = [re.compile(re.escape(k)) for k in kws]
    mask_pat = re.compile(re.escape(MASK_WORD))

    ids, t_orig, t_mask, labels, classes = load_pair(a.orig, a.masked)
    _, te = split_indices(labels, a.seed)
    if a.n_docs:
        te = te[: a.n_docs]
    models, tok = {}, None
    for cond in ("original", "masked"):
        path = os.path.join(a.models, f"{a.name}_{cond}_seed{a.seed}")
        models[cond] = AutoModelForSequenceClassification.from_pretrained(
            path, attn_implementation="eager").to(device).eval()
        tok = AutoTokenizer.from_pretrained(path)
    id2label = models["original"].config.id2label
    id2label = {int(k): v for k, v in id2label.items()}

    if a.plot_id:
        i = ids.index(a.plot_id)
        ro = analyse(models["original"], tok, t_orig[i], leak_pats, mask_pat, a.max_len, a.k, device)
        rm = analyse(models["masked"], tok, t_mask[i], leak_pats, mask_pat, a.max_len, a.k, device)
        plot(ro, rm, a.plot_id, labels[i], id2label,
             os.path.join(a.out, f"fig10_{a.plot_id.replace('/', '_')}.png"))
        return

    rows = []
    for i in te:
        ro = analyse(models["original"], tok, t_orig[i], leak_pats, mask_pat, a.max_len, a.k, device)
        rm = analyse(models["masked"], tok, t_mask[i], leak_pats, mask_pat, a.max_len, a.k, device)
        rows.append({
            "id": ids[i], "true": labels[i],
            "pred_orig": id2label[ro["pred"]], "pred_mask": id2label[rm["pred"]],
            "leak_topk_orig": ro["leak_in_topk"], "leak_topk_mask": rm["leak_in_topk"],
            "leak_mass_orig": ro["leak_mass"], "leak_mass_mask": rm["leak_mass"],
            "mask_token_mass": rm["mask_mass"],
        })
    df = pd.DataFrame(rows)
    df["both_correct"] = (df.pred_orig == df.true) & (df.pred_mask == df.true)
    df.to_csv(os.path.join(a.out, "attention_per_doc.csv"), index=False, encoding="utf-8")

    share_o = df.leak_topk_orig.mean() / a.k * 100
    share_m = df.leak_topk_mask.mean() / a.k * 100
    red = (1 - share_m / share_o) * 100 if share_o > 0 else float("nan")
    print(f"\nDocuments analysés : {len(df)}")
    print(f"Part des tokens de fuite dans le top-{a.k} : original {share_o:.2f} % -> masqué "
          f"{share_m:.2f} %  (réduction {red:.1f} %)")
    print(f"Masse d'attention sur la fuite : {df.leak_mass_orig.mean() * 100:.2f} % -> "
          f"{df.leak_mass_mask.mean() * 100:.2f} % ; sur [MASKED] : "
          f"{df.mask_token_mass.mean() * 100:.2f} %")

    cand = df[df.both_correct & (df.leak_topk_orig > 0)].sort_values("leak_mass_orig", ascending=False)
    print("\nMeilleurs candidats pour la Figure 10 (corrects dans les deux conditions) :")
    print(cand.head(10)[["id", "true", "leak_topk_orig", "leak_topk_mask",
                         "leak_mass_orig", "leak_mass_mask"]].to_string(index=False))


if __name__ == "__main__":
    main()
