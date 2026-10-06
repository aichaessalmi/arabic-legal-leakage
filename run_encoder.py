"""
Fine-tuning d'un encodeur BERT (AraBERTv2, CAMeLBERT-Mix) sur 5 seeds,
conditions original et masqué.

Différence importante avec le papier : le meilleur checkpoint est choisi sur
un jeu de VALIDATION (10 % du train), jamais sur le test.

Pour une RTX 2070 (8 Go) : batch 8 x accumulation 2 = batch effectif 16, fp16.

Exemples :
    python run_encoder.py --model aubmindlab/bert-base-arabertv02 --name arabertv2 \
        --orig data/original --masked data/masked --out results --keep-seed 42
    python run_encoder.py --model CAMeL-Lab/bert-base-arabic-camelbert-mix --name camelbert \
        --orig data/original --masked data/masked --out results
"""
import argparse
import gc
import os
import shutil

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from common import DEFAULT_SEEDS, load_pair, split_indices, save_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--orig", required=True)
    p.add_argument("--masked", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--conditions", nargs="+", default=["original", "masked"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--keep-seed", type=int, default=None,
                   help="Sauvegarde les modèles de ce seed (utile pour l'analyse d'attention)")
    p.add_argument("--no-normalize", action="store_true")
    a = p.parse_args()

    ids, t_orig, t_mask, labels, classes = load_pair(a.orig, a.masked, not a.no_normalize)
    label2id = {c: i for i, c in enumerate(classes)}
    y = np.array([label2id[l] for l in labels])
    ids = np.asarray(ids)
    tok = AutoTokenizer.from_pretrained(a.model)
    collator = DataCollatorWithPadding(tok)

    def make_ds(texts, idx):
        ds = Dataset.from_dict({"text": [texts[i] for i in idx], "label": y[idx].tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len),
                      batched=True, remove_columns=["text"])

    def metrics_fn(ev):
        preds = np.argmax(ev.predictions, axis=-1)
        return {"f1_macro": f1_score(ev.label_ids, preds, average="macro")}

    for seed in a.seeds:
        tr, va, te = split_indices(labels, seed, val_size=a.val_size)
        for cond in a.conditions:
            texts = t_orig if cond == "original" else t_mask
            set_seed(seed)
            cfg = AutoConfig.from_pretrained(a.model, num_labels=len(classes), classifier_dropout=0.3,
                                             label2id=label2id, id2label={i: c for c, i in label2id.items()})
            model = AutoModelForSequenceClassification.from_pretrained(a.model, config=cfg)
            run_dir = os.path.join(a.out, "tmp", f"{a.name}_{cond}_seed{seed}")
            args = TrainingArguments(
                output_dir=run_dir, num_train_epochs=a.epochs, learning_rate=a.lr,
                per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs * 2,
                gradient_accumulation_steps=a.grad_accum, warmup_ratio=0.1,
                lr_scheduler_type="linear", weight_decay=0.01,
                eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                load_best_model_at_end=True, metric_for_best_model="f1_macro",
                fp16=torch.cuda.is_available(), seed=seed, report_to="none", logging_steps=50,
            )
            trainer = Trainer(model=model, args=args, train_dataset=make_ds(texts, tr),
                              eval_dataset=make_ds(texts, va), data_collator=collator,
                              compute_metrics=metrics_fn)
            trainer.train()
            out = trainer.predict(make_ds(texts, te))
            pred = np.argmax(out.predictions, axis=-1)
            best_epoch = trainer.state.best_model_checkpoint
            save_run(a.out, a.name, cond, seed, ids[te].tolist(),
                     [classes[i] for i in y[te]], [classes[i] for i in pred],
                     extra={"best_checkpoint": best_epoch})

            if a.keep_seed is not None and seed == a.keep_seed:
                keep = os.path.join(a.out, "models", f"{a.name}_{cond}_seed{seed}")
                trainer.save_model(keep)
                tok.save_pretrained(keep)
            shutil.rmtree(run_dir, ignore_errors=True)
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
