#!/usr/bin/env python3
"""
ANZSRC dataset cleaning, step 8: labels that the text of a record does not support.

    python step8_labels.py <dataset_step7.csv> <output_dir>

Needs: pandas, numpy, scipy, scikit-learn >= 1.3. Takes about 20 minutes on 2 CPU cores.

1. Wrong journal codes (WRONG_CODES). In 14 cases every article of a journal carries a code from another
   field, e.g. 3601 "Art history, theory and criticism" on all of Journal of Biotechnology. The code is
   removed from ANZSRC_4D of that journal's records; the journal's other codes stay. Records left with
   no label move to disputed_labels.csv. (The title-only and non-English files contain none of these
   journal codes and are not touched by this step.)
2. Cross-validated check of what is left. TF-IDF of title + abstract, a one-vs-rest LinearSVC for each
   of the 213 groups, 5 folds grouped by journal / DOI prefix / repository host, so every record is
   scored by a model that has seen neither the record nor its source. Scores are calibrated per group
   (Platt scaling); a division's probability is that of its best group. A record moves to
   disputed_labels.csv when the model gives less than 5% to every division of its labels and at least
   60% to another division. Its labels are not changed; the model's suggestions are added as extra
   columns of the disputed file only.
3. Nothing else changes: ANZSRC (the source field) and all other columns stay as they are, rows keep
   their order, and the main file keeps its columns.
"""
import json, os, re, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')

# journal (Source title, case-insensitive) -> codes to remove; every article of the journal carries them
WRONG_CODES = {
    'journal of biotechnology': {'3601'},                          # Art history, theory and criticism
    'life': {'4601'},                                              # Applied computing
    'forensic science international synergy': {'3503'},            # Business systems in context
    'the linacre quarterly': {'4407', '4408'},                     # Policy and administration; Political science
    'quantitative imaging in medicine and surgery': {'5102'},      # Atomic, molecular and optical physics
    'currents in pharmacy teaching and learning': {'4010'},        # Engineering practice and education
    'structural dynamics': {'5107'},                               # Particle and high energy physics
    'literature and medicine': {'3904'},                           # Specialist studies in education
    'international journal of environmental health research': {'3505'},  # Human resources and industrial relations
    'biological trace element research': {'3102'},                 # Bioinformatics and computational biology
    'mathematical biosciences and engineering': {'4004'},          # Chemical engineering
    'journal of perianesthesia nursing': {'4204'},                 # Midwifery
    'tropical biomedicine': {'4208'},                              # Traditional, complementary and integrative medicine
}
STRONG_OWN, STRONG_OTHER = 0.05, 0.60   # division probability thresholds
TFIDF = dict(ngram_range=(1, 2), min_df=3, max_df=0.4, sublinear_tf=True, max_features=400000, stop_words='english',
             dtype=np.float32, token_pattern=r'(?u)\b[^\W\d_][\w\-]{1,}\b')


def fold_groups(d):
    """journal, else DOI prefix, else repository host, else the record itself"""
    pref = d.DOI.str.extract(r'^(10\.\d+/[A-Za-z0-9]+)', expand=False).fillna('').str.lower()
    host = d.Source_URLs.str.extract(r'https?://([^/;\s]+)', expand=False).fillna('')
    return np.where(d['Source title'] != '', 'J:' + d['Source title'].str.lower().str.strip(),
                    np.where(pref != '', 'D:' + pref, np.where(host != '', 'H:' + host, 'R:' + d.Record_ID)))


def check(d, log=print):
    """out-of-fold calibrated probabilities for every group"""
    t0 = time.time()
    X = TfidfVectorizer(**TFIDF).fit_transform((d.Title + ' . ' + d.Title + ' . ' + d.Abstract).tolist()).tocsr()
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(d.ANZSRC_4D.str.split(';')).astype(np.int32)  # int8 breaks OneVsRestClassifier
    groups = fold_groups(d)
    S = np.zeros(Y.shape, dtype=np.float32)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, Y[:, 0], groups)):
        m = OneVsRestClassifier(LinearSVC(C=0.5, random_state=0), n_jobs=2).fit(X[tr], Y[tr])
        S[te] = m.decision_function(X[te])
        log(f'  fold {k + 1}/5 done, {time.time() - t0:.0f}s')
    P = np.zeros_like(S)
    for c in range(Y.shape[1]):
        lr = LogisticRegression(C=100.0).fit(S[:, c:c + 1], Y[:, c])
        P[:, c] = lr.predict_proba(S[:, c:c + 1])[:, 1]
    return list(mlb.classes_), Y.astype(bool), P


def main(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    d = pd.read_csv(inp, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    cols = list(d.columns)

    # 1. wrong journal codes
    journal = d['Source title'].str.lower().str.strip()
    new_labels, removed = [], []
    for j, s in zip(journal, d.ANZSRC_4D):
        bad = WRONG_CODES.get(j, set())
        codes = s.split(';')
        new_labels.append(';'.join(c for c in codes if c not in bad))
        removed.append(';'.join(c for c in codes if c in bad))
    d['ANZSRC_4D'] = new_labels
    d['_removed'] = removed
    no_label = d.ANZSRC_4D.eq('')

    # 2. cross-validated check of the rest
    k = d[~no_label]
    print(f'check on {len(k)} records')
    cls, Y, P = check(k)
    div_of = np.array([c[:2] for c in cls]); divs = sorted(set(div_of))
    D = np.stack([P[:, div_of == v].max(1) for v in divs], 1)
    di = {v: n for n, v in enumerate(divs)}
    GD = np.zeros_like(D, dtype=bool)
    for n, s in enumerate(k.ANZSRC_4D):
        for c in s.split(';'):
            GD[n, di[c[:2]]] = True
    own = np.where(GD, D, -1).max(1)
    other = np.where(GD, -1, D).max(1)
    other_div = np.array(divs)[np.where(GD, -1, D).argmax(1)]
    strong = (own < STRONG_OWN) & (other >= STRONG_OTHER)
    top1 = P.argmax(1)
    metrics = {'top1_in_labels': float(Y[np.arange(len(Y)), top1].mean()),
               'top1_division_in_labels': float(GD[np.arange(len(Y)), [di[cls[t][:2]] for t in top1]].mean())}
    top3 = np.argsort(-P, 1)[:, :3]
    ci = {c: n for n, c in enumerate(cls)}
    chk = pd.DataFrame({
        'Record_ID': k.Record_ID.values, 'own_division_p': own, 'other_division': other_div, 'other_division_p': other,
        'strong': strong,
        'label_p': ['; '.join(f'{c} {P[n, ci[c]]:.2f}' for c in s.split(';')) for n, s in enumerate(k.ANZSRC_4D)],
        'model_top3': ['; '.join(f'{cls[t]} {P[n, t]:.2f}' for t in top3[n]) for n in range(len(k))]})

    # 3. outputs
    disputed_ids = set(k.Record_ID[strong]) | set(d.Record_ID[no_label])
    keep = ~d.Record_ID.isin(disputed_ids)
    d.loc[keep, cols].to_csv(os.path.join(outdir, 'dataset_step8.csv'), index=False, encoding='utf-8-sig')
    dis = d[~keep].merge(chk, on='Record_ID', how='left')
    dis['Disputed_reason'] = np.where(dis.ANZSRC_4D.eq(''), 'no_label_after_code_fix', 'text_supports_other_division')
    dis['Removed_codes'] = dis._removed
    dis['Model_label_probabilities'] = dis.label_p.fillna('')
    dis['Model_top3'] = dis.model_top3.fillna('')
    dis['Model_division'] = [f'{v} {p:.2f}' if isinstance(v, str) else '' for v, p in zip(dis.other_division, dis.other_division_p)]
    extra = ['Disputed_reason', 'Removed_codes', 'Model_label_probabilities', 'Model_top3', 'Model_division']
    dis[cols + extra].to_csv(os.path.join(outdir, 'disputed_labels.csv'), index=False, encoding='utf-8-sig')
    fixes = d[d._removed.ne('')][['Record_ID', 'Source title', '_removed', 'ANZSRC_4D']].rename(
        columns={'_removed': 'removed', 'ANZSRC_4D': 'after'})
    fixes.to_pickle(os.path.join(outdir, 'step8_code_fixes.pkl'))
    chk.to_pickle(os.path.join(outdir, 'step8_check.pkl'))
    s = {'main_in': len(d), 'records_with_code_fix': int(d._removed.ne('').sum()),
         'codes_removed': int(sum(len(x.split(';')) for x in d._removed if x)),
         'no_label_after_fix': int(no_label.sum()), 'checked': len(k), 'strong': int(strong.sum()),
         'disputed_total': int((~keep).sum()), 'main_out': int(keep.sum()), 'check_metrics': metrics}
    json.dump(s, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
