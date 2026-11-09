#!/usr/bin/env python3
"""
ANZSRC FoR 2020 classifier, step 4: combine the models and choose the decision rule on val.

    python 04_ensemble.py                       # every model found in models/
    python 04_ensemble.py --exclude scibert     # all but the named ones
    python 04_ensemble.py --models baseline specter2 roberta
    python 04_ensemble.py --models baseline_clean specter2_clean roberta_clean --name ensemble_clean
    python 04_ensemble.py --models baseline_clean specter2_clean roberta_clean modernbert_large_clean --name ensemble_large

Reads what steps 2 and 3 saved: models/<name>/val_probs.npy and test_probs.npy, and each model's own threshold
(models/baseline/threshold.json, models/<name>/best/config.json). Any further model saved the same way is
picked up too. Under a minute for two or three models, a few minutes for four or five, CPU only.

* Combination: weighted mean of the models' log-odds, each shifted so that the model's own threshold sits at 0.
  The two models scale their probabilities differently (thresholds 0.62 and 0.85), and the shift puts their
  decision points on a common footing. Weights are searched on a grid with step 0.1.
* Decision rule: one shared threshold, at least one group per record (per-group thresholds were tried and
  did not help: rare groups have too few examples in val to tune their own threshold).
* Weights and threshold are chosen by 5-fold cross-validation inside val, grouped by source: every val record
  is scored with a threshold tuned on the other folds, so the reported metrics are not tuned on the records
  they are measured on. The final threshold is tuned on all of val and applied to test in step 5.

Output: models/<name>/ (config.json, val_probs.npy, test_probs.npy), reports/04_<name>.json; <name> = ensemble
        unless --name says otherwise
"""
import argparse
import itertools
import json

import numpy as np

from common import (MODELS, REPORTS, _f1, evaluate, evaluate_by_origin, label_matrix, load_labels, load_split,
                    predict_sets, save_json, say, tune_threshold)

FOLDS = 5
STEP = 0.1
SEED = 0


def model_threshold(d):
    for f in (d / 'threshold.json', d / 'best' / 'config.json'):
        if f.exists():
            return float(json.loads(f.read_text(encoding='utf-8'))['threshold'])
    return None


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def combine(Z, w):
    return 1.0 / (1.0 + np.exp(-sum(wi * z for wi, z in zip(w, Z) if wi)))


def weight_grid(k):
    n = int(round(1 / STEP))
    return [np.array(c) / n for c in itertools.product(range(n + 1), repeat=k) if sum(c) == n]


def oof_thresholds(Y, P, fold):
    """every record gets the threshold tuned on the other folds"""
    thr = np.zeros(len(Y))
    for k in range(FOLDS):
        te = fold == k
        thr[te] = tune_threshold(Y[~te], P[~te])
    return thr


def oof(Y, P, fold, classes):
    """all metrics, out of fold"""
    thr = oof_thresholds(Y, P, fold)
    m, _ = evaluate(Y, P, thr[:, None], classes)
    return m, thr


def oof_score(Y, P, fold):
    """what the weights are chosen by: (micro-F1 + macro-F1) / 2 out of fold (only these two, so it is fast)"""
    micro, macro, _, _ = _f1(Y, predict_sets(P, oof_thresholds(Y, P, fold)[:, None]))
    micro, macro = round(micro, 4), round(macro, 4)
    return round((micro + macro) / 2, 4), micro, macro


def main():
    ap = argparse.ArgumentParser(description='Combine the models saved in models/')
    ap.add_argument('--models', nargs='+', help='use only these model folders')
    ap.add_argument('--exclude', nargs='+', default=[], help='leave these model folders out')
    ap.add_argument('--name', default='ensemble', help='output folder models/<name>')
    args = ap.parse_args()
    say('=== Крок 4: поєднання моделей і правило вибору груп ===\n')
    classes, _, _ = load_labels()
    va = load_split('val')
    Y = label_matrix(va.labels, classes)
    names, Zva, Zte = [], [], []
    for d in sorted(p for p in MODELS.iterdir() if p.is_dir() and not p.name.startswith('ensemble')):
        if (args.models and d.name not in args.models) or d.name in args.exclude:
            say(f'Модель {d.name}: не беру')
            continue
        thr = model_threshold(d)
        if thr is None or not (d / 'val_probs.npy').exists() or not (d / 'test_probs.npy').exists():
            continue
        Pv, Pt = np.load(d / 'val_probs.npy'), np.load(d / 'test_probs.npy')
        if Pv.shape != Y.shape:
            say(f'  {d.name}: інша кількість записів чи груп, пропускаю')
            continue
        names.append(d.name)
        Zva.append(logit(Pv) - logit(np.array(thr)))
        Zte.append(logit(Pt) - logit(np.array(thr)))
        say(f'Модель {d.name}: власний поріг {thr}')
    if len(names) < 2:
        raise SystemExit('Потрібні щонайменше дві моделі з кроків 2 і 3.')

    src = np.unique(va.source.values, return_inverse=True)[1]
    fold = np.random.RandomState(SEED).permutation(src.max() + 1)[src] % FOLDS
    say(f'\nПеребираю ваги для {len(names)} моделей, крос-валідація {FOLDS} частинами за джерелами...')
    rows = []
    for w in weight_grid(len(names)):
        score, micro, macro = oof_score(Y, combine(Zva, w), fold)
        rows.append({'weights': {n: round(float(x), 2) for n, x in zip(names, w)}, 'score': score,
                     'micro_f1': micro, 'macro_f1': macro})
    rows.sort(key=lambda r: -r['score'])
    w = np.array([rows[0]['weights'][n] for n in names])
    Pva, Pte = combine(Zva, w), combine(Zte, w)
    thr = tune_threshold(Y, Pva)

    out = MODELS / args.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'val_probs.npy', Pva.astype(np.float32))
    np.save(out / 'test_probs.npy', Pte.astype(np.float32))
    cfg = {'models': names, 'weights': rows[0]['weights'], 'threshold': thr, 'classes': classes,
           'rule': 'mean of log-odds shifted by each model\'s own threshold; one shared threshold; at least one group',
           'model_thresholds': {n: float(model_threshold(MODELS / n)) for n in names}}
    save_json(out / 'config.json', cfg)
    m_oof, thr_rows = oof(Y, Pva, fold, classes)
    best = {**rows[0], **m_oof}
    singles = {}
    for i, n in enumerate(names):
        one = np.eye(len(names))[i]
        singles[n] = {'weights': {k: float(x) for k, x in zip(names, one)}, **oof(Y, combine(Zva, one), fold, classes)[0]}
    by_origin = evaluate_by_origin(va, Y, Pva, thr_rows[:, None], classes)
    save_json(REPORTS / f'04_{args.name}.json', {'models': names, 'chosen': best, 'threshold_on_all_val': thr,
                                             'val_oof': m_oof, 'val_oof_by_origin': by_origin, 'singles': singles,
                                             'grid': rows, 'note': 'metrics are out-of-fold inside val'})

    say('\nМетрики на val (поріг щоразу підібрано на інших частинах val):')
    head = f'{"":36} {"P@1":>6} {"топ-3":>6} {"топ-5":>6} {"micro-F1":>9} {"macro-F1":>9} {"розділ P@1":>11}'
    say(head)
    for label, r in [(f'{n} окремо', singles[n]) for n in names] + [('разом ' + ', '.join(f'{n} {x:g}' for n, x in best['weights'].items() if x), best)]:
        say(f'{label:36} {r["p_at_1"]:6.3f} {r["hit_at_3"]:6.3f} {r["hit_at_5"]:6.3f} {r["micro_f1"]:9.3f} '
            f'{r["macro_f1"]:9.3f} {r["division_p_at_1"]:11.3f}')
    say('\nПоходження записів (разом):')
    for o, x in by_origin.items():
        say(f'  {o:13} ({x["records"]:5}): P@1 {x["p_at_1"]:.3f}, топ-3 {x["hit_at_3"]:.3f}, micro-F1 {x["micro_f1"]:.3f}')
    say(f'\nОстаточний поріг (на всьому val): {thr}; міток на запис {m_oof["labels_pred"]:.2f} (справжніх {m_oof["labels_true"]:.2f})')
    say(f'Збережено: {out}')
    say(f'Звіт: {REPORTS / f"04_{args.name}.json"}')


if __name__ == '__main__':
    main()
