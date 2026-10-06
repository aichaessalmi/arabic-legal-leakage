# -*- coding: utf-8 -*-
"""
Crée les fichiers publics (sans texte ni données personnelles) :
  release/document_ids.csv      : identifiants publics de chaque décision
  release/split_random.csv      : train/val/test pour les 5 seeds (même logique que common.py)

    python make_release_files.py
"""
import os, glob, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "CORPUS_PROCESSED")
OUT = os.path.join(HERE, "release")
os.makedirs(OUT, exist_ok=True)
EN = {"الغرفة الإجتماعية": "Social", "الغرفة الإدارية": "Administrative", "الغرفة التجارية": "Commercial",
      "الغرفة الجنائية": "Criminal", "الغرفة العقارية": "Real Estate", "الغرفة المدنية": "Civil",
      "غرفة الأحوال الشخصية و الميراث": "Personal Status"}

rows = []
for cls in sorted(os.listdir(BASE)):
    d = os.path.join(BASE, cls)
    if not os.path.isdir(d):
        continue
    for f in sorted(glob.glob(os.path.join(d, "*_cleaned.txt"))):
        name = os.path.basename(f)
        meta = {}
        j = os.path.join(d, name.replace("_cleaned.txt", ".json"))
        if os.path.exists(j):
            try:
                with open(j, encoding="utf-8") as fh:
                    meta = json.load(fh).get("METADATA", {}).get("CASSATION_LEVEL", {}) or {}
            except Exception:
                meta = {}
        rows.append({"id": f"{cls}/{name}", "chamber_ar": cls, "chamber_en": EN.get(cls, cls),
                     "decision_number": meta.get("decision_number"),
                     "decision_date": meta.get("decision_date"),
                     "case_file_number": meta.get("dossier_number")})
df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
df.to_csv(os.path.join(OUT, "document_ids.csv"), index=False, encoding="utf-8-sig")
print(f"document_ids.csv : {len(df)} décisions | métadonnées trouvées : "
      f"{df.decision_number.notna().mean():.1%} (numéro), {df.decision_date.notna().mean():.1%} (date)")

y = df.chamber_ar.values
idx = np.arange(len(df))
split = pd.DataFrame({"id": df.id})
for seed in (42, 13, 7, 2024, 123):
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    tr, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=seed)
    s = np.empty(len(df), dtype=object); s[tr] = "train"; s[va] = "val"; s[te] = "test"
    split[f"seed{seed}"] = s
split.to_csv(os.path.join(OUT, "split_random.csv"), index=False, encoding="utf-8-sig")
print("split_random.csv :", split.seed42.value_counts().to_dict())
