#!/usr/bin/env python3
"""
ANZSRC dataset cleaning, step 1: remove duplicates.

    python step1_dedup.py <input.csv> <output_dir> [--extra extra.csv] [--pairs pairs.pkl]

A duplicate is another copy of the same publication/record, or a record whose
title and abstract are identical to another one. Tiers, applied in order; each
tier only looks at records that survived the previous tiers:

  T1  same DOI
  T2  same DOI apart from a version suffix (.v1, .v2 ... figshare/Zenodo style)
  T3  same title and same abstract
  T4  same abstract (>=100 normalised chars) and one title is the other title
      extended (subtitle, "[Summary]", "Supplementary materials for ...")
  T5  same title (>=30 normalised chars), no abstract in either record
  T6  near-identical text (TF-IDF cosine of title+abstract >= 0.90, both
      abstracts >=100 chars) and titles equal or one contained in the other
  T4 and T6 skip abstracts that appear under >= 3 different titles: such a text is a
  series description or boilerplate and does not show that two records are one work.

Not removed, only reported: different titles sharing one identical abstract
(series with a common description, boilerplate text, wrongly attached abstracts).

Which record of a duplicate group is kept (first criterion wins):
  1. real labels before AI-model labels (Label_origin/Record_origin mention
     an LLM, Claude, ChatGPT, a consensus of AI reviewers, predicted labels, ...)
  2. record with a usable abstract
  3. main publication before erratum/corrigendum/supplementary-material records
  4. Record_origin: original / original_updated > new > new_material > other
  5. longer abstract
  6. has a DOI
  7. more filled-in metadata fields
  8. Record_ID (stable tie-break)

Normalisation: HTML unescaped, tags removed, Unicode NFKD, case-folded, every
non-alphanumeric character dropped (letters of any script are kept).
Kept records are written unchanged, in their original order.
"""
import argparse, html, os, re, time, unicodedata, difflib
import numpy as np, pandas as pd

NEAR_COS = 0.90
MIN_ABS = 100          # normalised characters
MIN_CONTAINED = 20     # shortest title allowed to count as "contained" in another
BOILER_TITLES = 3      # identical abstract under >= this many different titles = series/boilerplate text

AI_PAT = re.compile(r'(?:llm|gpt|claude|gemini|consensus|reviewer|ai[_\- ]?label|model[_\- ]?label|'
                    r'predicted|auto[_\- ]?label|machine[_\- ]?label|ai[_\- ]?generated)', re.I)
MINOR_PAT = re.compile(r'^\W*(?:erratum|corrigendum|correction|retraction|supplementary|supplemental)', re.I)
ORIGIN_RANK = {'original': 0, 'original_updated': 0, 'new': 1, 'new_material': 2}

TAG = re.compile(r'<[^>]+>')
NONWORD = re.compile(r'[\W_]+')

REASONS = {
    'T1_same_DOI': 'Той самий DOI',
    'T2_DOI_version': 'Той самий DOI, інша версія (.v1, .v2 …)',
    'T3_same_title_abstract': 'Однакові назва й анотація',
    'T4_same_abstract_title_variant': 'Однакова анотація, назва — варіант тієї самої (підзаголовок, [Summary], Supplementary …)',
    'T5_same_title_no_abstract': 'Однакова назва, анотацій немає в обох',
    'T6_near_duplicate': 'Майже ідентичний текст (≥0.90) і та сама назва — інша редакція того самого запису',
}


def norm_text(x):
    x = TAG.sub(' ', html.unescape(str(x)))
    return NONWORD.sub('', unicodedata.normalize('NFKD', x).casefold())


def norm_doi(x):
    x = str(x).strip().lower()
    x = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', x)
    x = re.sub(r'^doi:\s*', '', x)
    return x if x.startswith('10.') else ''


def title_variant(a, b):
    if a == b:
        return True
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    return len(s) >= MIN_CONTAINED and s in l


def codes(s):
    return frozenset(c for c in re.split(r'[;,\s]+', str(s)) if c)


def label_relation(removed, kept):
    a, b = codes(removed), codes(kept)
    if a == b: return 'identical'
    if not a or not b: return 'one_empty'
    if a < b: return 'removed_subset_of_kept'
    if a > b: return 'removed_superset_of_kept'
    if a & b: return 'partial_overlap'
    return 'disjoint'


def build_keys(d, cols):
    d['_doi'] = d.DOI.map(norm_doi)
    d['_doib'] = d._doi.str.replace(r'\.v\d+$', '', regex=True)
    d['_t'] = d.Title.map(norm_text)
    d['_a'] = d.Abstract.map(norm_text)
    d['_alen'] = d._a.str.len()
    d['_ai'] = (d.Label_origin.str.contains(AI_PAT) | d.Record_origin.str.contains(AI_PAT)).astype(int)
    d['_noabs'] = (~d.Has_usable_abstract.astype(str).str.strip().isin(['1', 'True', 'true', 'yes'])).astype(int)
    d['_minor'] = d.Title.str.contains(MINOR_PAT).astype(int)
    d['_orank'] = d.Record_origin.map(ORIGIN_RANK).fillna(3).astype(int)
    d['_nodoi'] = (d._doi == '').astype(int)
    d['_filled'] = (d[cols] != '').sum(axis=1)
    order = d.sort_values(['_ai', '_noabs', '_minor', '_orank', '_alen', '_nodoi', '_filled', 'Record_ID'],
                          ascending=[True, True, True, True, False, True, False, True]).index
    d['_rank'] = pd.Series(np.arange(len(d)), index=order).reindex(d.index)
    return d


def near_pairs(d, idx, thr):
    """pairs (row labels) with TF-IDF cosine >= thr on title + abstract"""
    import gc
    from sklearn.feature_extraction.text import TfidfVectorizer
    clean = lambda x: TAG.sub(' ', html.unescape(str(x)))
    txt = [clean(t) + ' . ' + clean(a) for t, a in zip(d.Title.values[idx], d.Abstract.values[idx])]
    vec = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32,
                          token_pattern=r'(?u)\b[^\W\d_][\w\-]{2,}\b')
    X = vec.fit_transform(txt).tocsr()
    del txt, vec; gc.collect()
    XT = X.T.tocsr()
    R, C, V = [], [], []
    B = 1000
    for s in range(0, X.shape[0], B):
        S = (X[s:s + B] @ XT).tocoo()
        m = (S.data >= thr) & (S.col > S.row + s)
        R.append(S.row[m] + s); C.append(S.col[m]); V.append(S.data[m])
        del S, m
    ids = np.asarray(idx)
    return pd.DataFrame({'i': ids[np.concatenate(R)], 'j': ids[np.concatenate(C)], 'cos': np.concatenate(V)})


def leader_cluster(pairs, rank):
    """greedy: best-ranked record keeps all its still-unassigned neighbours"""
    nb = {}
    for i, j, c in pairs:
        nb.setdefault(i, []).append((j, c)); nb.setdefault(j, []).append((i, c))
    done, out = set(), []
    for i in sorted(nb, key=lambda x: rank[x]):
        if i in done: continue
        done.add(i)
        for j, c in sorted(nb[i], key=lambda x: rank[x[0]]):
            if j not in done:
                done.add(j); out.append((j, i, c))
    return out


def run(d, cols, pairs=None, log_fn=print):
    t0 = time.time()
    d = build_keys(d.reset_index(drop=True), cols)
    log_fn(f'rows {len(d)}; AI-labelled rows {int(d._ai.sum())}')
    alive = pd.Series(True, index=d.index)
    rank = d._rank.values
    log = []

    def exact_tier(code, key, mask):
        sub = d[alive & mask]
        sub = sub.assign(_k=key[sub.index]).sort_values('_rank')
        dup = sub.duplicated('_k', keep='first')
        kept = pd.Series(sub[~dup].index, index=sub[~dup]._k)
        for i, k in zip(sub[dup].index, sub[dup]._k):
            log.append((i, kept[k], code, np.nan))
        alive[sub[dup].index] = False
        log_fn(f'{code}: {int(dup.sum())}')

    exact_tier('T1_same_DOI', d._doi, d._doi != '')
    exact_tier('T2_DOI_version', d._doib, d._doib != '')
    exact_tier('T3_same_title_abstract', d._t + '|' + d._a, (d._t != '') & (d._alen > 0))

    # an abstract carried by >= 3 records with different titles is a series description or
    # boilerplate; it says nothing about whether two of those records are the same publication
    sub = d[alive & (d._alen >= MIN_ABS)]
    ntitles = sub.groupby('_a')._t.nunique()
    boiler = set(ntitles[ntitles >= BOILER_TITLES].index)
    d['_boiler'] = d._a.isin(boiler)
    log_fn(f'boilerplate/series abstracts: {len(boiler)} texts, {int((alive & d._boiler).sum())} records')

    # T4: identical abstract, one title extends the other
    sub = d[alive & (d._alen >= MIN_ABS) & ~d._boiler]
    grp = sub.groupby('_a').filter(lambda g: len(g) > 1)
    cand = []
    for _, g in grp.groupby('_a'):
        ix = g.index.tolist(); ts = g._t.tolist()
        for x in range(len(ix)):
            for y in range(x + 1, len(ix)):
                if title_variant(ts[x], ts[y]):
                    cand.append((ix[x], ix[y], np.nan))
    for j, i, c in leader_cluster(cand, rank):
        alive[j] = False; log.append((j, i, 'T4_same_abstract_title_variant', c))
    log_fn(f'T4_same_abstract_title_variant: {sum(1 for r in log if r[2].startswith("T4"))}')

    exact_tier('T5_same_title_no_abstract', d._t, (d._t.str.len() >= 30) & (d._alen == 0))

    # T6: near-identical text with the same (or contained) title
    idx = d.index[alive]
    P = pairs if pairs is not None else near_pairs(d, idx, NEAR_COS)
    P = P[(P.cos >= NEAR_COS) & alive.values[P.i] & alive.values[P.j]]
    P = P[(d._alen.values[P.i] >= MIN_ABS) & (d._alen.values[P.j] >= MIN_ABS)]
    P = P[~d._boiler.values[P.i] & ~d._boiler.values[P.j]]
    ti = d._t.values
    cand = [(i, j, c) for i, j, c in zip(P.i, P.j, P.cos) if title_variant(ti[i], ti[j])]
    n6 = 0
    for j, i, c in leader_cluster(cand, rank):
        alive[j] = False; log.append((j, i, 'T6_near_duplicate', c)); n6 += 1
    log_fn(f'T6_near_duplicate: {n6}   ({time.time()-t0:.0f}s)')

    # report only: identical abstract shared by records with different titles
    sub = d[alive & (d._alen >= MIN_ABS)]
    shared = sub[sub.duplicated('_a', keep=False)].copy()
    shared['group'] = shared.groupby('_a').ngroup() + 1
    return d, alive, log, shared


def make_log(d, cols, log):
    L = pd.DataFrame(log, columns=['ri', 'ki', 'reason_code', 'text_cosine'])
    R, K = d.loc[L.ri].reset_index(drop=True), d.loc[L.ki].reset_index(drop=True)
    out = pd.DataFrame({
        'reason_code': L.reason_code, 'reason': L.reason_code.map(REASONS),
        'removed_Record_ID': R.Record_ID, 'kept_Record_ID': K.Record_ID,
        'removed_is_AI_labelled': R._ai.astype(bool), 'kept_is_AI_labelled': K._ai.astype(bool),
        'removed_Record_origin': R.Record_origin, 'kept_Record_origin': K.Record_origin,
        'removed_ANZSRC_4D': R.ANZSRC_4D, 'kept_ANZSRC_4D': K.ANZSRC_4D,
        'label_relation': [label_relation(a, b) for a, b in zip(R.ANZSRC_4D, K.ANZSRC_4D)],
        'title_similarity': [round(difflib.SequenceMatcher(None, a[:300], b[:300]).ratio(), 3) for a, b in zip(R._t, K._t)],
        'text_cosine': L.text_cosine.astype(float).round(3),
        'removed_DOI': R.DOI, 'kept_DOI': K.DOI, 'removed_Title': R.Title, 'kept_Title': K.Title,
    })
    return pd.concat([out, R[cols].add_prefix('removed: ')], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('outdir')
    ap.add_argument('--extra', help='extra CSV with the same columns (e.g. AI-labelled records) appended before dedup')
    ap.add_argument('--pairs', help='precomputed TF-IDF pairs for the input (pickle with i, j, cos)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    d = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    cols = list(d.columns)
    if args.extra:
        e = pd.read_csv(args.extra, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        d = pd.concat([d, e.reindex(columns=cols, fill_value='')], ignore_index=True)
    pairs = pd.read_pickle(args.pairs) if (args.pairs and not args.extra) else None
    d, alive, log, shared = run(d, cols, pairs)
    d.loc[alive, cols].to_csv(os.path.join(args.outdir, 'dataset_step1_dedup.csv'), index=False, encoding='utf-8-sig')
    make_log(d, cols, log).to_pickle(os.path.join(args.outdir, 'removed_log.pkl'))
    shared.to_pickle(os.path.join(args.outdir, 'shared_abstracts.pkl'))
    import json
    json.dump({'rows_in': len(d), 'ai_labelled_rows': int(d._ai.sum()), 'kept': int(alive.sum()),
               'removed': len(log), 'removed_ai_labelled': int(d._ai.values[[r[0] for r in log]].sum()) if log else 0,
               'shared_records': len(shared), 'shared_groups': int(shared.group.max()) if len(shared) else 0},
              open(os.path.join(args.outdir, 'summary.json'), 'w'), indent=1)
    print(f'kept {int(alive.sum())}  removed {len(log)}  shared-abstract records kept: {len(shared)}')


if __name__ == '__main__':
    main()
