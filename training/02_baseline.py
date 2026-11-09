#!/usr/bin/env python3
"""
ANZSRC FoR 2020 classifier, step 2: baseline model, TF-IDF + logistic regression.

    python 02_baseline.py
    python 02_baseline.py --train train_clean        # on the cleaned train (step 6) -> models/baseline_clean

Needs data/ from step 1 and common.py next to this file. About 10-20 minutes on a 6-core laptop.

* Features: TF-IDF of words and word pairs in title (twice) + abstract, 200 000 features.
* One logistic regression per group (213, one-vs-rest), class_weight='balanced' so that rare groups count;
  C (inverse regularisation strength) = 32.
* One shared threshold for "the record has this group", tuned on val for micro-F1; every record gets at
  least its most probable group.
* Metrics are computed on val only. Test probabilities are saved for step 5 and not scored here.

Output: models/<name>/ (vectorizer.joblib, coef.npy, intercept.npy, val_probs.npy, test_probs.npy,
        threshold.json) and reports/02_<name>.json; <name> is baseline, or baseline_clean for --train train_clean
"""
import argparse
import os
import time

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import (MODELS, REPORTS, evaluate, evaluate_by_origin, label_matrix, load_labels, load_split,
                    save_json, say, show, tfidf_text, tune_threshold)

C = 32
TFIDF = dict(ngram_range=(1, 2), min_df=3, max_df=0.4, sublinear_tf=True, max_features=200_000,
             stop_words='english', dtype=np.float32, token_pattern=r'(?u)\b[^\W\d_][\w\-]{1,}\b')
N_JOBS = max(1, (os.cpu_count() or 2) // 2)


def fit_one(X, y):
    m = LogisticRegression(C=C, solver='liblinear', class_weight='balanced', max_iter=1000)
    m.fit(X, y)
    return m.coef_.ravel().astype(np.float32), np.float32(m.intercept_[0])


def probs(X, W, b):
    z = X @ W.T + b
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description='TF-IDF + logistic regression baseline')
    ap.add_argument('--train', default='train', help='training part in data/ (train or train_clean)')
    ap.add_argument('--name', default=None, help='output folder models/<name>')
    args = ap.parse_args()
    name = args.name or ('baseline' if args.train == 'train' else 'baseline_' + args.train.replace('train_', ''))
    OUT = MODELS / name
    t0 = time.time()
    say(f'=== Крок 2: базова модель TF-IDF + логістична регресія ({args.train} -> models/{name}) ===\n')
    classes, names, _ = load_labels()
    tr, va, te = load_split(args.train), load_split('val'), load_split('test')
    Ytr, Yva = label_matrix(tr.labels, classes), label_matrix(va.labels, classes)
    say(f'train {len(tr)}, val {len(va)}, test {len(te)} записів; {len(classes)} груп')

    say('Будую ознаки TF-IDF...')
    vec = TfidfVectorizer(**TFIDF)
    Xtr = vec.fit_transform(tfidf_text(tr))
    Xtr.sum_duplicates()          # canonical form: the parallel workers get a read-only copy that
    Xtr.sort_indices()            # scikit-learn would otherwise try to sort in place
    Xva, Xte = vec.transform(tfidf_text(va)), vec.transform(tfidf_text(te))
    say(f'  {Xtr.shape[1]} ознак, {time.time() - t0:.0f} с')

    say(f'Навчаю {len(classes)} класифікаторів (C = {C}, процесів {N_JOBS})...')
    t1 = time.time()
    W = np.zeros((len(classes), Xtr.shape[1]), dtype=np.float32)
    b = np.zeros(len(classes), dtype=np.float32)
    jobs = Parallel(n_jobs=N_JOBS, return_as='generator')(delayed(fit_one)(Xtr, Ytr[:, j]) for j in range(len(classes)))
    step = max(1, len(classes) // 10)
    for j, (w, c) in enumerate(jobs):
        W[j], b[j] = w, c
        if (j + 1) % step == 0 or j + 1 == len(classes):
            say(f'  {j + 1}/{len(classes)}, {time.time() - t1:.0f} с')

    Pva, Pte = probs(Xva, W, b), probs(Xte, W, b)
    thr = tune_threshold(Yva, Pva)
    m, per = evaluate(Yva, Pva, thr, classes)
    by_origin = evaluate_by_origin(va, Yva, Pva, thr, classes)

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, OUT / 'vectorizer.joblib')
    np.save(OUT / 'coef.npy', W.astype(np.float16))
    np.save(OUT / 'intercept.npy', b)
    np.save(OUT / 'val_probs.npy', Pva)
    np.save(OUT / 'test_probs.npy', Pte)
    save_json(OUT / 'threshold.json', {'threshold': thr, 'classes': classes})
    per_group = sorted(({'code': c, 'name': names.get(c, ''), 'train': int(Ytr[:, j].sum()), 'val': int(Yva[:, j].sum()),
                         'f1': None if np.isnan(per[j]) else round(float(per[j]), 4)} for j, c in enumerate(classes)),
                       key=lambda g: (g['f1'] is None, g['f1'] if g['f1'] is not None else 0))
    save_json(REPORTS / f'02_{name}.json', {
        'train': args.train, 'model': 'tfidf + one-vs-rest logistic regression', 'C': C,
        'tfidf': {k: str(v) for k, v in TFIDF.items()}, 'features': int(Xtr.shape[1]), 'threshold': thr,
        'val': m, 'val_by_origin': by_origin, 'val_per_group': per_group, 'minutes': round((time.time() - t0) / 60, 1)})

    say(f'\nПоріг (спільний для всіх груп): {thr}')
    show('val', m)
    for o, x in by_origin.items():
        show(f'  {o} ({x["records"]})', x)
    say(f'Міток на запис: справжніх {m["labels_true"]:.2f}, передбачених {m["labels_pred"]:.2f}')
    worst = [g for g in per_group if g['f1'] is not None][:5]
    say('Найслабші групи (F1 на val): ' + '; '.join(f'{g["code"]} {g["f1"]:.2f}' for g in worst))
    say(f'\nМодель: {OUT}')
    say(f'Звіт: {REPORTS / f"02_{name}.json"}  ({(time.time() - t0) / 60:.1f} хв)')


if __name__ == '__main__':
    main()
