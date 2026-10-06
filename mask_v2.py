# -*- coding: utf-8 -*-
"""
Masquage v2 : complète le masquage v1 (DATASET_MASKED) en neutralisant les
fuites résiduelles trouvées par l'audit :
  1. numéros de dossier / de décision  (ex. 2019/2/5/3039 : le 3e chiffre code la chambre)
  2. années                            (répartition très différente selon les chambres)
  3. composition de la formation        (noms du président de chambre et des conseillers)
  4. variantes non masquées de « الملف X » (ex. OCR « الملف المرني », « الملف الشرعى »)
  5. juridictions / formations d'origine qui trahissent la chambre
     (المحكمة الإدارية، محكمة الاستئناف التجارية، غرفة الجنح الاستئنافية ...)

Deux corpus sont produits à partir de DATASET_MASKED :
  DATASET_MASKED_V2  : v1 + masquages ci-dessus
  DATASET_BODY_V3    : v2 sans la première ligne (titre) -> contenu juridique uniquement

    python mask_v2.py            (from the repository root)
"""
import os, re, glob, json

NUM, YEAR, COURT, MASK = " [NUM] ", " [YEAR] ", " [COURT] ", "[MASKED]"

def _flex(word):
    """Motif tolérant aux variantes orthographiques arabes."""
    out = []
    for ch in word:
        if ch in "اأإآ": out.append("[اأإآ]")
        elif ch in "ةه": out.append("[ةه]")
        elif ch in "يى": out.append("[يى]")
        elif ch == " ": out.append(r"\s+")
        else: out.append(re.escape(ch))
    return "".join(out)

COURTS = [
    "محكمة الاستئناف الإدارية", "محاكم الاستئناف الإدارية", "المحكمة الإدارية", "المحاكم الإدارية",
    "محكمة الاستئناف التجارية", "محاكم الاستئناف التجارية", "المحكمة التجارية", "المحاكم التجارية",
    "غرفة الجنح الاستئنافية", "غرفة الجنايات الاستئنافية", "غرفة الجنايات الابتدائية",
    "غرفة الاستئنافات الجنحية", "الغرفة الجنحية", "غرفة الجنح", "غرفة الجنايات",
    "القسم الإداري", "القسم المدني", "القسم التجاري", "القسم الجنحي",
]
COURT_RE = re.compile("|".join(_flex(c) for c in sorted(COURTS, key=len, reverse=True)))
FILE_RE  = re.compile(r"\d+(?:\s*/\s*\d+)+")                 # 2019/2/5/3039, 1/244, 12/01/2022
YEAR_RE  = re.compile(r"(?<!\d)(?:19[5-9]\d|20[0-3]\d)(?!\d)")
DOSSIER_RE = re.compile(r"(الملف|ملف)\s+(?!\[MASKED\])(\S+)(\s+(?:عدد|رقم))")   # « الملف المرني عدد »
COMPO_KW = re.compile(r"رئيس[ةا]?\s+الغرفة|رئيسا|المستشار(?:ة)?\s+المقرر|مقرر[اة]|المستشارين|المستشارون|"
                      r"كاتب(?:ة)?\s+الضبط|المحامي\s+العام|الهيئة\s+الحاكمة|أعضاء|عضوا")
FOOTER_RE = re.compile(r"وبه\s+صدر\s+القرار|وكانت\s+الهيئة\s+الحاكمة|الهيئة\s+الحاكمة\s+متركبة")


SEG_RE = re.compile(r"[^.\n؛]*[.\n؛]?")


def _drop_composition(text):
    """Supprime les phrases (courtes) qui décrivent la composition de la formation ;
    dans une phrase longue, retire seulement une fenêtre autour du mot-clé."""
    out = []
    for seg in SEG_RE.findall(text):
        if not seg or not COMPO_KW.search(seg):
            out.append(seg)
        elif len(seg) < 400:
            out.append("\n" if seg.endswith("\n") else " ")
        else:
            out.append(re.sub(r"\S+(?:\s+\S+){0,5}\s*(?:" + COMPO_KW.pattern + r")(?:\s+\S+){0,6}", " ", seg))
    return "".join(out)


def mask_v2(text):
    # 3a. couper le pied de page (composition) s'il est dans la seconde moitié du texte
    for mt in FOOTER_RE.finditer(text):
        if mt.start() > len(text) * 0.5:
            text = text[:mt.start()]
            break
    text = _drop_composition(text)                                             # 3b
    text = DOSSIER_RE.sub(lambda m: f"{m.group(1)} {MASK}{m.group(3)}", text)   # 4
    text = COURT_RE.sub(COURT, text)                                           # 5
    text = FILE_RE.sub(NUM, text)                                              # 1
    text = YEAR_RE.sub(YEAR, text)                                             # 2
    return re.sub(r"[ \t]+", " ", text).strip()


def body_v3(text):
    """v2 sans l'en-tête (titre avec numéro de décision, date, numéro de dossier)."""
    t = mask_v2(text)
    nl = t.find("\n")
    if 0 < nl < 400:
        return t[nl + 1:].strip()
    k = t.find("[NUM]", 0, 400)
    return t[k + 5:].strip() if k >= 0 else t


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = here  # run from the repository root
    src = os.path.join(root, "DATASET_MASKED")
    for name, fn in (("DATASET_MASKED_V2", mask_v2), ("DATASET_BODY_V3", body_v3)):
        dst = os.path.join(root, name); n = 0
        for cls in sorted(os.listdir(src)):
            if not os.path.isdir(os.path.join(src, cls)):
                continue
            os.makedirs(os.path.join(dst, cls), exist_ok=True)
            for f in glob.glob(os.path.join(src, cls, "*_cleaned.txt")):
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    t = fn(fh.read())
                with open(os.path.join(dst, cls, os.path.basename(f)), "w", encoding="utf-8") as fh:
                    fh.write(t)
                n += 1
        print(f"{name} : {n} documents écrits dans {dst}")
