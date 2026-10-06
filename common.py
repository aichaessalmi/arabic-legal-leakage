"""
Module commun pour toutes les expériences.

Structure de données attendue (celle de l'Algorithme 1 du papier) :
    data/original/<chambre>/<fichier>.txt
    data/masked/<chambre>/<fichier>.txt
Les deux corpus doivent contenir exactement les mêmes noms de fichiers.
"""
import os
import re
import glob
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_SEEDS = [42, 13, 7, 2024, 123]

# ---------------------------------------------------------------------------
# Normalisation arabe : appliquée IDENTIQUEMENT aux deux corpus.
# (Dans le papier, la normalisation n'était appliquée qu'au corpus original :
#  cela introduit une deuxième différence entre les conditions, en plus du
#  masquage. Ici, la seule différence restante est le masquage.)
# ---------------------------------------------------------------------------
_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")
_HTML = re.compile(r"<[^>]+>")
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")
MASK_WORD = "MASKED"


def normalize(text: str) -> str:
    text = text.replace("[MASKED]", f" {MASK_WORD} ")  # protège le jeton de masque
    text = _HTML.sub(" ", text)
    text = _TASHKEEL.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def _load_dir(root):
    docs = {}
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.txt"))):
            with open(f, encoding="utf-8") as fh:
                docs[f"{label}/{os.path.basename(f)}"] = (fh.read(), label)
    return docs


def load_pair(orig_root, masked_root, do_normalize=True):
    """Retourne ids, textes_orig, textes_masques, labels (str), noms_de_classes."""
    orig, masked = _load_dir(orig_root), _load_dir(masked_root)
    missing = set(orig) ^ set(masked)
    if missing:
        raise ValueError(f"{len(missing)} fichiers non appariés entre les deux corpus, "
                         f"ex. : {sorted(missing)[:5]}")
    ids = sorted(orig)
    prep = normalize if do_normalize else (lambda t: t)
    t_orig = [prep(orig[i][0]) for i in ids]
    t_mask = [prep(masked[i][0]) for i in ids]
    labels = [orig[i][1] for i in ids]
    class_names = sorted(set(labels))
    return ids, t_orig, t_mask, labels, class_names


# ---------------------------------------------------------------------------
# Splits : même split pour original et masqué, pour un seed donné
# ---------------------------------------------------------------------------
def split_indices(labels, seed, test_size=0.2, val_size=0.0):
    """Split stratifié. Si val_size > 0, une partie du TRAIN sert de validation
    (sélection du meilleur checkpoint) : le test n'est JAMAIS utilisé pour choisir."""
    idx = np.arange(len(labels))
    y = np.asarray(labels)
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    if val_size > 0:
        tr, va = train_test_split(tr, test_size=val_size, stratify=y[tr], random_state=seed)
        return tr, va, te
    return tr, te


# ---------------------------------------------------------------------------
# Métriques et sauvegarde
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def save_run(out_dir, model, condition, seed, test_ids, y_true, y_pred, extra=None):
    """Sauvegarde les prédictions (pour McNemar) et les métriques (pour l'agrégation)."""
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    tag = f"{model}__{condition}__seed{seed}"
    pd.DataFrame({"id": test_ids, "y_true": y_true, "y_pred": y_pred}).to_csv(
        os.path.join(out_dir, "predictions", f"{tag}.csv"), index=False, encoding="utf-8")
    m = compute_metrics(y_true, y_pred)
    m.update({"model": model, "condition": condition, "seed": seed, "n_test": len(y_true)})
    if extra:
        m.update(extra)
    with open(os.path.join(out_dir, "metrics", f"{tag}.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    print(f"[{tag}] acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")
    return m
