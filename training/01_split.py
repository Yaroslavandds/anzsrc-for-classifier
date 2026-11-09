#!/usr/bin/env python3
"""
ANZSRC FoR 2020 classifier, step 1: split the data into train / val / test (80 / 10 / 10).

    python 01_split.py

Input:  ../anzsrc4_cleaned.csv; ../anzsrc2020_for.xlsx (names of groups and divisions, optional)
Output: data/train.parquet, data/val.parquet, data/test.parquet, data/labels.json, reports/01_split.json

The split is by source, so that val and test show how the model does on journals and repositories it has
not seen in training:
* a record's source is its journal (Source title); without one, the DOI prefix (10.6084/m9 = figshare);
  without that, the repository host from Source_URLs; otherwise the record itself (the same rule as in the
  label check of step 8);
* records with the same title (>= 30 normalised characters) always stay in one part;
* a source goes to one part as a whole. Exception: sources with more than LARGE records whose articles carry
  many different label sets (the most common set covers less than HOMOGENEOUS of them, e.g. Scientific
  Reports, PLOS ONE). One of them would otherwise fill half of the test, and with labels set article by
  article there is no journal-wide label to learn, so they are split record by record. A large source with
  mostly identical label sets goes to train as a whole;
* the parts are stratified by the 213 groups: iterative stratification (Sechidis et al., 2011) applied to
  whole sources. The group with the fewest unplaced records is handled first; each source goes to the part
  that most lacks that group, provided the part still has room for it.
Every run gives the same split (ties are broken with a fixed seed).
"""
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'anzsrc4_cleaned.csv'
NAMES = ROOT.parent / 'anzsrc2020_for.xlsx'
OUT = ROOT / 'data'
REPORT = ROOT / 'reports' / '01_split.json'
SPLITS = ('train', 'val', 'test')
RATIOS = np.array([0.8, 0.1, 0.1])
LARGE = 500            # sources bigger than this may be split record by record ...
HOMOGENEOUS = 0.25     # ... unless their most common label set covers at least this share of their records
MAX_EVAL_UNIT = 600    # a unit (source / title cluster) bigger than this always goes to train
SLACK = 0.002          # val / test may overshoot their size by this share of all records
SEED = 42


def say(*a):
    print(*a, flush=True)


def source_keys(d):
    pref = d.DOI.str.lower().str.extract(r'^(10\.\d+/[a-z0-9]+)', expand=False).fillna('')
    host = d.Source_URLs.str.extract(r'https?://([^/;\s]+)', expand=False).fillna('').str.lower()
    journal = d['Source title'].str.lower().str.strip()
    return pd.Series(np.where(journal != '', 'J:' + journal,
                              np.where(pref != '', 'D:' + pref,
                                       np.where(host != '', 'H:' + host, 'R:' + d.Record_ID))), index=d.index)


def norm_title(t):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKD', t).casefold())


def find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def union_all(parent, idx):
    r0 = find(parent, idx[0])
    for i in idx[1:]:
        r = find(parent, i)
        if r != r0:
            parent[r] = r0


def group_names():
    """{'3001': 'Agricultural biotechnology', ...}, {'30': 'Agricultural, veterinary and food sciences', ...}"""
    if not NAMES.exists():
        return {}, {}
    try:
        t = pd.read_excel(NAMES, sheet_name='Table 2', header=None, dtype=str).iloc[:, :3].fillna('')
    except Exception:
        return {}, {}
    grp, div = {}, {}
    for a, b, c in t.itertuples(index=False):
        a, b, c = a.strip(), b.strip(), c.strip()
        if re.fullmatch(r'\d{2}', a) and b:
            div[a] = b.capitalize()
        if re.fullmatch(r'\d{4}', b) and c:
            grp[b] = c
    return grp, div


def stratify(C, n, forced, rng):
    """iterative stratification of units. C: units x labels counts, n: unit sizes. Returns part index per unit."""
    need_l = np.outer(RATIOS, C.sum(0)).astype(float)
    need_n = RATIOS * n.sum()
    slack = SLACK * n.sum()
    part = np.full(len(n), -1)
    for u in np.where(forced)[0]:
        part[u] = 0
        need_l[0] -= C[u]
        need_n[0] -= n[u]
    order = rng.permutation(len(n))           # fixed random order of units within a label
    free = part < 0
    rem = C[free].sum(0).astype(float)
    while free.any():
        labs = np.where(rem > 0)[0]
        if len(labs) == 0:                     # cannot happen: every record has a label
            units = np.where(free)[0]
        else:
            lab = labs[np.argmin(rem[labs])]
            units = np.where(free & (C[:, lab] > 0))[0]
        for u in units[np.argsort(order[units])]:
            ok = [s for s in range(len(RATIOS)) if s == 0 or n[u] <= need_n[s] + slack]
            key = (lambda s: (need_l[s, lab], need_n[s], rng.rand())) if len(labs) else (lambda s: (need_n[s], rng.rand()))
            s = max(ok, key=key)
            part[u] = s
            need_l[s] -= C[u]
            need_n[s] -= n[u]
            free[u] = False
            rem -= C[u]
    return part


def refine(C, n, part, forced, rng, passes=6, slack=0.005):
    """local search after stratification: move single units between parts while that brings every group's
    counts closer to 80/10/10 (chi-square distance) and val / test stay within +-slack of their size"""
    k = len(RATIOS)
    target = np.outer(RATIOS, C.sum(0)) + 1e-9
    w = 1.0 / target
    A = np.stack([C[part == s].sum(0) for s in range(k)]).astype(float)
    size = np.array([n[part == s].sum() for s in range(k)], dtype=float)
    lo, hi = (RATIOS - slack) * n.sum(), (RATIOS + slack) * n.sum()
    movable = np.where(~forced)[0]
    for _ in range(passes):
        moved = 0
        for u in movable[rng.permutation(len(movable))]:
            s1 = part[u]
            nz = np.nonzero(C[u])[0]
            cv = C[u, nz]
            out1 = (((A[s1, nz] - cv - target[s1, nz]) ** 2 - (A[s1, nz] - target[s1, nz]) ** 2) * w[s1, nz]).sum()
            best, best_s = 0.0, -1
            for s2 in range(k):
                if s2 == s1 or (s2 > 0 and size[s2] + n[u] > hi[s2]) or (s1 > 0 and size[s1] - n[u] < lo[s1]):
                    continue
                delta = out1 + (((A[s2, nz] + cv - target[s2, nz]) ** 2 - (A[s2, nz] - target[s2, nz]) ** 2) * w[s2, nz]).sum()
                if delta < best - 1e-9:
                    best, best_s = delta, s2
            if best_s >= 0:
                A[s1, nz] -= cv
                A[best_s, nz] += cv
                size[s1] -= n[u]
                size[best_s] += n[u]
                part[u] = best_s
                moved += 1
        if not moved:
            break
    return part


def main():
    t0 = time.time()
    say('=== Крок 1: розбиття на train / val / test ===\n')
    d = pd.read_csv(DATA, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    say(f'Прочитано {len(d):,} записів з {DATA.name}'.replace(',', ' '))
    classes = sorted({c for s in d.ANZSRC_4D for c in s.split(';') if c})
    ci = {c: k for k, c in enumerate(classes)}
    Y = np.zeros((len(d), len(classes)), dtype=np.int32)
    for r, s in enumerate(d.ANZSRC_4D):
        for c in s.split(';'):
            Y[r, ci[c]] = 1

    # sources, and which of them are split record by record
    src = source_keys(d)
    size = src.value_counts()
    top_share = d.groupby(src).ANZSRC_4D.agg(lambda s: s.value_counts().iloc[0] / len(s))
    big = size[size > LARGE].index
    by_record = sorted(s for s in big if top_share[s] < HOMOGENEOUS)
    train_only = sorted(s for s in big if top_share[s] >= HOMOGENEOUS)

    # units: whole sources (except by_record) joined with records that share a title
    parent = list(range(len(d)))
    by_record_set = set(by_record)
    for s, idx in d.groupby(src).indices.items():
        if s not in by_record_set and len(idx) > 1:
            union_all(parent, list(idx))
    nt = d.Title.map(norm_title)
    for t, idx in d.groupby(nt).indices.items():
        if len(t) >= 30 and len(idx) > 1:
            union_all(parent, list(idx))
    root = np.array([find(parent, i) for i in range(len(d))])
    unit = pd.factorize(root, sort=True)[0]
    n_units = unit.max() + 1
    n = np.bincount(unit, minlength=n_units)
    C = np.zeros((n_units, len(classes)), dtype=np.int64)
    np.add.at(C, unit, Y)
    forced = np.zeros(n_units, dtype=bool)
    forced[np.unique(unit[src.isin(train_only).values])] = True
    forced |= n > MAX_EVAL_UNIT

    rng = np.random.RandomState(SEED)
    part_of_unit = refine(C, n, stratify(C, n, forced, rng), forced, rng)
    part = part_of_unit[unit]

    # outputs
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    cols = {'Record_ID': 'id', 'Title': 'title', 'Abstract': 'abstract', 'ANZSRC_4D': 'labels',
            'Record_origin': 'record_origin', 'Material_group': 'material_group', 'Publication_type': 'publication_type',
            'Source title': 'source_title', 'DOI': 'doi', 'Year': 'year'}
    out = d[list(cols)].rename(columns=cols)
    out['source'] = src.values
    for k, name in enumerate(SPLITS):
        out[part == k].reset_index(drop=True).to_parquet(OUT / f'{name}.parquet', index=False)

    grp, div = group_names()
    counts = {name: Y[part == k].sum(0) for k, name in enumerate(SPLITS)}
    labels = {'groups': [{'code': c, 'name': grp.get(c, ''), 'division': c[:2],
                          **{name: int(counts[name][j]) for name in SPLITS}} for j, c in enumerate(classes)],
              'divisions': [{'code': v, 'name': div.get(v, '')} for v in sorted({c[:2] for c in classes})]}
    (OUT / 'labels.json').write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding='utf-8')

    # report
    total = Y.sum(0)
    rep = {'input': {'path': str(DATA), 'records': len(d), 'groups': len(classes)},
           'params': {'ratios': RATIOS.tolist(), 'large_source': LARGE, 'homogeneous_share': HOMOGENEOUS,
                      'max_eval_unit': MAX_EVAL_UNIT, 'slack': SLACK, 'seed': SEED},
           'sources': {'total': int(len(size)),
                       'split_by_record': [{'source': s, 'records': int(size[s]), 'top_label_set_share': round(float(top_share[s]), 3)}
                                           for s in by_record],
                       'train_only': [{'source': s, 'records': int(size[s]), 'top_label_set_share': round(float(top_share[s]), 3)}
                                      for s in train_only]},
           'units': int(n_units), 'splits': {}, 'origin_share': {}, 'rare_groups': [], 'checks': {}}
    for k, name in enumerate(SPLITS):
        m = part == k
        rep['splits'][name] = {'records': int(m.sum()), 'share': round(float(m.mean()), 4),
                               'sources': int(src[m].nunique()), 'groups_present': int((counts[name] > 0).sum()),
                               'labels_per_record': round(float(Y[m].sum() / m.sum()), 3)}
        rep['origin_share'][name] = {o: round(float(v), 3) for o, v in d.Record_origin[m].value_counts(normalize=True).items()}
    big_cls = total >= 100
    dev = {name: counts[name] / total - RATIOS[k] for k, name in enumerate(SPLITS)}
    rep['class_balance'] = {name: {'max_abs_dev_pp': round(float(np.abs(dev[name][big_cls]).max() * 100), 2),
                                   'mean_abs_dev_pp': round(float(np.abs(dev[name][big_cls]).mean() * 100), 2)}
                            for name in SPLITS}
    rep['class_balance']['note'] = 'deviation of each group\'s share from 80/10/10, groups with >= 100 records'
    for j in np.argsort(total)[:15]:
        rep['rare_groups'].append({'code': classes[j], 'name': grp.get(classes[j], ''), 'total': int(total[j]),
                                   **{name: int(counts[name][j]) for name in SPLITS}})
    rep['groups_missing'] = {name: [classes[j] for j in np.where(counts[name] == 0)[0]] for name in SPLITS}
    tparts = pd.DataFrame({'t': nt, 'p': part})[nt.str.len() >= 30].groupby('t').p.nunique()
    dois = pd.DataFrame({'doi': d.DOI.str.lower(), 'p': part})[d.DOI != ''].groupby('doi').p.nunique()
    sparts = pd.DataFrame({'s': src, 'p': part})[~src.isin(by_record_set)].groupby('s').p.nunique()
    rep['checks'] = {'titles_in_two_parts': int((tparts > 1).sum()), 'dois_in_two_parts': int((dois > 1).sum()),
                     'whole_sources_in_two_parts': int((sparts > 1).sum())}
    rep['seconds'] = round(time.time() - t0, 1)
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')

    # summary
    fmt = lambda x: f'{x:,}'.replace(',', ' ')
    say(f'Джерел (журналів, репозиторіїв): {fmt(len(size))}')
    biggest = sorted(by_record, key=lambda s: -size[s])
    say(f'  ділимо по записах {len(by_record)} великих джерел з різними мітками: '
        + ', '.join(f'{s[2:]} ({fmt(size[s])})' for s in biggest[:4]) + (' …' if len(by_record) > 4 else ''))
    for s in train_only:
        say(f'  цілком у train: {s[2:]} ({fmt(size[s])} записів, найчастіший набір міток у {top_share[s]:.0%})')
    say('')
    for name in SPLITS:
        x = rep['splits'][name]
        say(f'{name:5}: {fmt(x["records"]):>7} записів ({x["share"]:.1%}), джерел {fmt(x["sources"])}, '
            f'груп {x["groups_present"]} з {len(classes)}, міток на запис {x["labels_per_record"]}')
    say(f'\nЧастки груп (з ≥ 100 записів) відхиляються від 80/10/10 у середньому на '
        + ', '.join(f'{name} {rep["class_balance"][name]["mean_abs_dev_pp"]} п.п.' for name in SPLITS))
    say('Найменші групи (train / val / test): ' + '; '.join(
        f'{g["code"]}: {g["train"]}/{g["val"]}/{g["test"]}' for g in rep['rare_groups'][:8]))
    for name in ('val', 'test'):
        if rep['groups_missing'][name]:
            say(f'У {name} немає груп: {", ".join(rep["groups_missing"][name])}')
    c = rep['checks']
    say(f'Перевірки: однакові назви в різних частинах {c["titles_in_two_parts"]}, однакові DOI {c["dois_in_two_parts"]}, '
        f'цілі джерела в кількох частинах {c["whole_sources_in_two_parts"]}')
    say(f'\nФайли: {OUT}\\{{train,val,test}}.parquet, labels.json' if sys.platform == 'win32' else
        f'\nФайли: {OUT}/{{train,val,test}}.parquet, labels.json')
    say(f'Звіт: {REPORT}  ({rep["seconds"]} с)')


if __name__ == '__main__':
    main()
