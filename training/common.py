"""Shared helpers for the ANZSRC FoR 2020 training scripts: paths, data, label matrices, thresholds, metrics."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
SPLITS = ('train', 'val', 'test')
ORIGINS = ('original', 'new', 'new_material')       # original_updated counts as original


def say(*a):
    print(*a, flush=True)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')


def load_labels():
    """classes in the fixed order of step 1, {code: name}, {division: name}"""
    L = json.loads((DATA / 'labels.json').read_text(encoding='utf-8'))
    return [g['code'] for g in L['groups']], {g['code']: g['name'] for g in L['groups']}, \
        {v['code']: v['name'] for v in L['divisions']}


def load_split(name):
    return pd.read_parquet(DATA / f'{name}.parquet')


def label_matrix(labels, classes):
    ci = {c: k for k, c in enumerate(classes)}
    Y = np.zeros((len(labels), len(classes)), dtype=np.uint8)
    for r, s in enumerate(labels):
        for c in s.split(';'):
            if c:
                Y[r, ci[c]] = 1
    return Y


def tfidf_text(df):
    """title twice + abstract (the title weighs more)"""
    return (df.title + ' . ' + df.title + ' . ' + df.abstract).tolist()


def predict_sets(P, thr):
    """groups whose probability reaches the threshold (one shared value or one per group);
    a record always gets at least its most probable group"""
    pred = P >= thr
    pred[np.arange(len(P)), P.argmax(1)] = True
    return pred


def _f1(Y, pred):
    Y, pred = Y.astype(bool), pred.astype(bool)
    tp, fp, fn = (Y & pred).sum(0), (~Y & pred).sum(0), (Y & ~pred).sum(0)
    micro = 2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum())
    den = 2 * tp + fp + fn
    per = np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan)
    samples = float(np.mean(2 * (Y & pred).sum(1) / np.maximum(1, Y.sum(1) + pred.sum(1))))
    return float(micro), float(np.nanmean(per)), samples, per


def divisions_of(M, classes):
    divs = sorted({c[:2] for c in classes})
    col = np.array([divs.index(c[:2]) for c in classes])
    return np.stack([M[:, col == k].any(1) for k in range(len(divs))], 1), col


def tune_threshold(Y, P, grid=None):
    """one shared threshold with the best micro-F1 (the most probable group of a record always counts as chosen).
    The counts for all thresholds come from one histogram: the same result as trying the thresholds one by one,
    tens of times faster"""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else np.sort(np.asarray(grid, dtype=np.float64))
    Y = np.asarray(Y).astype(bool)
    rows, t1 = np.arange(len(P)), P.argmax(1)
    top = np.zeros(P.shape, dtype=bool)
    top[rows, t1] = True
    pos_top = int(Y[rows, t1].sum())
    edges = np.concatenate([[-np.inf], grid, [np.inf]])
    at_least = lambda v: np.cumsum(np.histogram(v, bins=edges)[0][::-1])[::-1][1:len(grid) + 1]   # count >= grid[k]
    tp = pos_top + at_least(P[~top & Y])
    fp = (len(P) - pos_top) + at_least(P[~top & ~Y])
    fn = int(Y.sum()) - tp
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    return float(grid[int(np.argmax(f1))])


def evaluate(Y, P, thr, classes):
    """metrics dict and per-group F1"""
    from sklearn.metrics import average_precision_score
    pred = predict_sets(P, thr)
    micro, macro, samples, per = _f1(Y, pred)
    top = np.argsort(-P, 1)[:, :3]
    D, col = divisions_of(Y, classes)
    Dp, _ = divisions_of(pred, classes)
    has = Y.sum(0) > 0
    top5 = np.argsort(-P, 1)[:, :5]
    Pd = np.stack([P[:, col == k].max(1) for k in range(D.shape[1])], 1)      # a division = its best group
    dtop = np.argsort(-Pd, 1)[:, :2]
    # lenient P@1: a label "Other ..." (code ending in 99) also accepts any group of its own division,
    # since such a label only says that the work is in that division but outside its named groups
    other = np.array([c.endswith('99') for c in classes])
    Do = np.stack([Y[:, (col == k) & other].any(1) for k in range(D.shape[1])], 1)
    rows, t1 = np.arange(len(Y)), top[:, 0]
    m = {'micro_f1': micro, 'macro_f1': macro, 'samples_f1': samples,
         'p_at_1': float(Y[rows, t1].mean()),
         'p_at_1_other_ok': float((Y[rows, t1].astype(bool) | Do[rows, col[t1]]).mean()),
         'hit_at_3': float((np.take_along_axis(Y, top, 1).sum(1) > 0).mean()),
         'hit_at_5': float((np.take_along_axis(Y, top5, 1).sum(1) > 0).mean()),
         'recall_at_3': float(np.take_along_axis(Y, top, 1).sum() / Y.sum()),
         'division_micro_f1': _f1(D, Dp)[0],
         'division_p_at_1': float(D[np.arange(len(Y)), col[top[:, 0]]].mean()),
         'division_hit_at_2': float(np.take_along_axis(D, dtop, 1).any(1).mean()),
         'macro_ap': float(average_precision_score(Y[:, has], P[:, has], average='macro')),
         'labels_true': float(Y.sum(1).mean()), 'labels_pred': float(pred.sum(1).mean())}
    return {k: round(v, 4) for k, v in m.items()}, per


def evaluate_by_origin(df, Y, P, thr, classes):
    origin = df.record_origin.replace({'original_updated': 'original'}).values
    thr = np.asarray(thr)
    per_row = thr.ndim > 0 and thr.shape[0] == len(Y) and thr.shape[0] != Y.shape[1]
    out = {}
    for o in ORIGINS:
        m = origin == o
        if m.sum():
            out[o] = {'records': int(m.sum()), **evaluate(Y[m], P[m], thr[m] if per_row else thr, classes)[0]}
    return out


def show(title, m):
    say(f'{title}: micro-F1 {m["micro_f1"]:.3f} | macro-F1 {m["macro_f1"]:.3f} | P@1 {m["p_at_1"]:.3f} | '
        f'R@3 {m["recall_at_3"]:.3f} | розділи: micro-F1 {m["division_micro_f1"]:.3f}, P@1 {m["division_p_at_1"]:.3f}')
