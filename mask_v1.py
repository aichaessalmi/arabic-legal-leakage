# -*- coding: utf-8 -*-
"""
Script de masquage des indices de classe pour le dataset juridique marocain.
Remplace toutes les mentions de chambres et types de dossiers par [MASKED].
Génère un dataset masqué + rapport de statistiques.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
ORIGINAL_PATH = "CORPUS_PROCESSED"   # run from the repository root
MASKED_OUTPUT = "DATASET_MASKED"

MASK_TOKEN = "[MASKED]"

# ─────────────────────────────────────────────
# TERMES À MASQUER
# Chaque groupe couvre les variantes orthographiques :
# - avec/sans hamza : الإجتماعية / الاجتماعية
# - masculin/féminin : الاجتماعي / الاجتماعية
# ─────────────────────────────────────────────

# Racines des adjectifs de classe (couvre toutes les formes)
CLASS_ADJECTIVES = [
    # Sociale
    r"[اإ]لا?[اإ]جتماعي[ةه]?",
    r"ال[اإ]جتماعي[ةه]?",
    # Administrative
    r"ال[اإ]داري[ةه]?",
    # Commerciale
    r"التجاري[ةه]?",
    # Pénale/Criminelle
    r"الجنائي[ةه]?",
    r"الجنح[ةي]?",
    r"الجنحي[ةه]?",
    # Immobilière
    r"العقاري[ةه]?",
    # Civile
    r"المدني[ةه]?",
    # Statut personnel et héritage
    r"ال[اأ]حوال الشخصي[ةه]",
    r"الشخصي[ةه]",
    r"الم[يى]راث",
    r"ال[اإ]رث",
]

# Expressions complètes fréquentes (masquées en priorité, avant les adjectifs seuls)
FULL_EXPRESSIONS = [
    # ── Chambres ──
    r"[بال]*غرف[ةه]\s+ال[اإ]جتماعي[ةه]",
    r"[بال]*غرف[ةه]\s+ال[اإ]داري[ةه]",
    r"[بال]*غرف[ةه]\s+التجاري[ةه]",
    r"[بال]*غرف[ةه]\s+الجنائي[ةه]",
    r"[بال]*غرف[ةه]\s+العقاري[ةه]",
    r"[بال]*غرف[ةه]\s+المدني[ةه]",
    r"[بال]*غرف[ةه]\s+ال[اأ]حوال\s+الشخصي[ةه](\s+و\s*الم[يى]راث)?",
    # ── Types de dossiers ──
    r"الملف\s+ال[اإ]جتماعي",
    r"الملف\s+ال[اإ]داري",
    r"الملف\s+التجاري",
    r"الملف\s+الجنائي",
    r"الملف\s+الجنحي",
    r"الملف\s+العقاري",
    r"الملف\s+المدني",
    r"الملف\s+الشرعي",
    r"ملف\s+ال[اأ]حوال\s+الشخصي[ةه]",
    # ── Types de قضية/قضايا ──
    r"القضايا\s+ال[اإ]جتماعي[ةه]",
    r"القضايا\s+ال[اإ]داري[ةه]",
    r"القضايا\s+التجاري[ةه]",
    r"القضايا\s+الجنائي[ةه]",
    r"القضايا\s+العقاري[ةه]",
    r"القضايا\s+المدني[ةه]",
    r"القضي[ةه]\s+ال[اإ]جتماعي[ةه]",
    r"القضي[ةه]\s+ال[اإ]داري[ةه]",
    r"القضي[ةه]\s+التجاري[ةه]",
    r"القضي[ةه]\s+الجنائي[ةه]",
    r"القضي[ةه]\s+العقاري[ةه]",
    r"القضي[ةه]\s+المدني[ةه]",
    # ── Matières ──
    r"الماد[ةه]\s+ال[اإ]جتماعي[ةه]",
    r"الماد[ةه]\s+ال[اإ]داري[ةه]",
    r"الماد[ةه]\s+التجاري[ةه]",
    r"الماد[ةه]\s+الجنائي[ةه]",
    r"الماد[ةه]\s+العقاري[ةه]",
    r"الماد[ةه]\s+المدني[ةه]",
]

# Compiler les regex (une seule fois pour la performance)
COMPILED_FULL = [re.compile(p) for p in FULL_EXPRESSIONS]

# ─────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────
def mask_text(text):
    """
    Masque les indices de classe dans le texte.
    Retourne le texte masqué + le nombre de remplacements + les termes trouvés.
    """
    n_replacements = 0
    terms_found = Counter()
    
    # Étape 1 : masquer les expressions complètes
    for pattern in COMPILED_FULL:
        matches = pattern.findall(text)
        if matches:
            for m in pattern.finditer(text):
                terms_found[m.group(0)] += 1
            text, n = pattern.subn(MASK_TOKEN, text)
            n_replacements += n
    
    return text, n_replacements, terms_found


def process_dataset():
    """Traite tout le dataset et génère la version masquée."""
    base = Path(ORIGINAL_PATH)
    output = Path(MASKED_OUTPUT)
    output.mkdir(exist_ok=True)
    
    if not base.exists():
        print(f"❌ Chemin introuvable : {ORIGINAL_PATH}")
        return
    
    global_stats = {}
    global_terms = Counter()
    total_files = 0
    total_replacements = 0
    
    print("🔄 Début du masquage...\n")
    
    for class_dir in sorted(base.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        out_class_dir = output / class_name
        out_class_dir.mkdir(exist_ok=True)
        
        class_replacements = 0
        class_terms = Counter()
        n_files = 0
        
        txt_files = list(class_dir.glob("*_cleaned.txt"))
        
        for f in txt_files:
            try:
                text = f.read_text(encoding="utf-8")
            except:
                text = f.read_text(encoding="utf-8-sig", errors="ignore")
            
            masked, n_repl, terms = mask_text(text)
            
            # Sauvegarder le fichier masqué
            out_file = out_class_dir / f.name
            out_file.write_text(masked, encoding="utf-8")
            
            class_replacements += n_repl
            class_terms.update(terms)
            n_files += 1
        
        total_files += n_files
        total_replacements += class_replacements
        global_terms.update(class_terms)
        
        global_stats[class_name] = {
            "files": n_files,
            "replacements": class_replacements,
            "avg_per_doc": class_replacements / n_files if n_files > 0 else 0,
            "top_terms": dict(class_terms.most_common(10))
        }
        
        print(f"📁 {class_name}")
        print(f"   Fichiers traités    : {n_files}")
        print(f"   Termes masqués      : {class_replacements}")
        print(f"   Moyenne par doc     : {class_replacements/n_files:.1f}" if n_files else "")
        top3 = class_terms.most_common(3)
        for term, cnt in top3:
            print(f"   → '{term}' : {cnt} fois")
        print()
    
    # ─── Rapport global ───
    print("=" * 60)
    print(f"  RÉSUMÉ GLOBAL")
    print("=" * 60)
    print(f"Total fichiers        : {total_files}")
    print(f"Total termes masqués  : {total_replacements}")
    print(f"Moyenne par document  : {total_replacements/total_files:.1f}")
    
    print(f"\n🔝 Top 20 des termes les plus masqués :")
    for term, cnt in global_terms.most_common(20):
        print(f"   {cnt:>6} × '{term}'")
    
    # Sauvegarder le rapport JSON
    report = {
        "total_files": total_files,
        "total_replacements": total_replacements,
        "mask_token": MASK_TOKEN,
        "per_class": global_stats,
        "top_terms_global": dict(global_terms.most_common(50))
    }
    
    report_path = output / "masking_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Rapport sauvegardé : {report_path}")
    print(f"✅ Dataset masqué dans : {output.resolve()}")


# ─────────────────────────────────────────────
# VÉRIFICATION SUR UN ÉCHANTILLON
# ─────────────────────────────────────────────
def test_sample():
    """Teste le masquage sur un exemple avant de traiter tout le dataset."""
    sample = """قرار محكمة النقض رقم 31 الصادر بتاريخ 12 يناير 2022 في الملف الاجتماعي رقم 2019/2/5/3039
باسم جلالة الملك وطبقا للقانون
وكانت الهيئة الحاكمة متركبة من رئيسة الغرفة الاجتماعية السيدة مليكة بنزاهير
حيث تقدمت المطلوبة في القضية الاجتماعية بمقال أمام الغرفة الإدارية"""
    
    masked, n, terms = mask_text(sample)
    print("═" * 60)
    print("TEST SUR ÉCHANTILLON")
    print("═" * 60)
    print(f"\nAVANT :\n{sample}")
    print(f"\nAPRÈS :\n{masked}")
    print(f"\nRemplacements : {n}")
    print(f"Termes : {dict(terms)}")
    print("═" * 60)


if __name__ == "__main__":
    # 1. D'abord tester sur un échantillon
    test_sample()
    
    # 2. Demander confirmation
    answer = input("\n▶ Lancer le masquage complet ? (o/n) : ")
    if answer.lower() in ["o", "oui", "y", "yes"]:
        process_dataset()
    else:
        print("Masquage annulé.")