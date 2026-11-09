#!/usr/bin/env python3
"""
ANZSRC FoR 2020 classifier, step 5: the one-time evaluation on test.

    python 05_evaluate.py
    python 05_evaluate.py --ensemble ensemble_clean --compare baseline specter2 ensemble
                                         # the models retrained on cleaned labels next to the earlier ones

Uses what steps 2-4 saved: every model's test probabilities and own threshold, and the combination chosen in
step 4 (models/<ensemble>/config.json). Nothing is tuned here: weights and thresholds were fixed on val, so these
are the final, honest numbers for journals and repositories the models have never seen. Runs in about a minute.

Output: reports/05_test.json and reports/05_test_report.xlsx (for another ensemble: 05_test_<suffix>.*)
        (summary, results by record origin, every group, the most frequent confusions, 100 example errors).
"""
import argparse
import json

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from common import (DATA, MODELS, REPORTS, evaluate, evaluate_by_origin, label_matrix, load_labels,
                    load_split, predict_sets, save_json, say)

NAMES = {'baseline': 'TF-IDF', 'specter2': 'SPECTER2', 'scibert': 'SciBERT', 'roberta': 'RoBERTa',
         'roberta_large': 'RoBERTa-large', 'modernbert_large': 'ModernBERT-large', 'deberta_large': 'DeBERTa-v3-large',
         'ensemble': 'Поєднання',
         'ensemble_large': 'Поєднання 5 моделей', 'ensemble_final': 'Підсумкове поєднання'}


def nice(name):
    base, clean = (name[:-6], True) if name.endswith('_clean') else (name, False)
    return NAMES.get(base, base) + (' (чисті мітки)' if clean else '')

METRICS = [('p_at_1', 'Перша група правильна (P@1)'),
           ('p_at_1_other_ok', 'P@1 для «Інше …»: годиться розділ'),
           ('hit_at_3', 'Правильна група серед перших трьох'),
           ('hit_at_5', 'Правильна група серед перших п\'яти'), ('micro_f1', 'micro-F1'), ('macro_f1', 'macro-F1'),
           ('samples_f1', 'samples-F1'), ('macro_ap', 'macro-AP (без порогу)'),
           ('division_p_at_1', 'Перший розділ правильний'), ('division_hit_at_2', 'Правильний розділ серед перших двох'),
           ('division_micro_f1', 'Розділи: micro-F1'), ('labels_pred', 'Міток на запис (поставлено)'),
           ('labels_true', 'Міток на запис (справжніх)')]


def model_threshold(d):
    for f in (d / 'threshold.json', d / 'best' / 'config.json', d / 'config.json'):     # the last: an ensemble
        if f.exists():
            return float(json.loads(f.read_text(encoding='utf-8'))['threshold'])
    return None


def main():
    ap = argparse.ArgumentParser(description='Final evaluation on test')
    ap.add_argument('--ensemble', default='ensemble', help='folder models/<ensemble> from step 4')
    ap.add_argument('--compare', nargs='*', default=[],
                    help='earlier models or ensembles to show next to it, e.g. baseline specter2 ensemble')
    args = ap.parse_args()
    suffix = '' if args.ensemble == 'ensemble' else '_' + args.ensemble.replace('ensemble_', '')
    say(f'=== Крок 5: остаточна оцінка на test ({args.ensemble}) ===\n')
    classes, names, divn = load_labels()
    ens = json.loads((MODELS / args.ensemble / 'config.json').read_text(encoding='utf-8'))
    clean = args.ensemble.endswith('_clean') or any(m.endswith('_clean') for m in ens['models'])
    train_part = 'train_clean' if clean and (DATA / 'train_clean.parquet').exists() else 'train'
    tr, te = load_split(train_part), load_split('test')
    Y = label_matrix(te.labels, classes)
    n_train = label_matrix(tr.labels, classes).sum(0)
    if ens['classes'] != classes:
        raise SystemExit('Порядок груп у моделях і в data/labels.json різний: перезапусти крок 4.')

    results = {}
    for name in ens['models']:
        results[name], _ = evaluate(Y, np.load(MODELS / name / 'test_probs.npy'), model_threshold(MODELS / name), classes)
    P = np.load(MODELS / args.ensemble / 'test_probs.npy')
    thr = ens['threshold']
    m, per = evaluate(Y, P, thr, classes)
    results[args.ensemble] = m
    cols = list(ens['models']) + [args.ensemble]
    for name in args.compare:
        d = MODELS / name
        t = model_threshold(d)
        if t is None or not (d / 'test_probs.npy').exists():
            say(f'Для порівняння немає {d}, пропускаю')
            continue
        results['було:' + name], _ = evaluate(Y, np.load(d / 'test_probs.npy'), t, classes)
        cols.append('було:' + name)
    label = lambda c: 'Було: ' + nice(c[5:]) if c.startswith('було:') else nice(c)
    by_origin = evaluate_by_origin(te, Y, P, thr, classes)
    val = json.loads((REPORTS / f'04_{args.ensemble}.json').read_text(encoding='utf-8'))['val_oof']

    # every group
    pred = predict_sets(P, thr)
    Yb, pb = Y.astype(bool), pred.astype(bool)
    tp, fp, fn = (Yb & pb).sum(0), (~Yb & pb).sum(0), (Yb & ~pb).sum(0)
    from sklearn.metrics import average_precision_score
    groups = []
    for j, c in enumerate(classes):
        ap = average_precision_score(Y[:, j], P[:, j]) if Y[:, j].any() else None
        groups.append({'code': c, 'name': names.get(c, ''), 'division': c[:2], 'train': int(n_train[j]),
                       'test': int(Y[:, j].sum()), 'predicted': int(pb[:, j].sum()),
                       'precision': float(tp[j] / (tp[j] + fp[j])) if tp[j] + fp[j] else None,
                       'recall': float(tp[j] / (tp[j] + fn[j])) if tp[j] + fn[j] else None,
                       'f1': None if np.isnan(per[j]) else float(per[j]), 'ap': ap})

    # confusions: the first group is not a label -> (label, first group) pairs
    top = P.argmax(1)
    wrong = np.where(Y[np.arange(len(Y)), top] == 0)[0]
    pairs = {}
    for i in wrong:
        for j in np.where(Y[i])[0]:
            key = (classes[j], classes[top[i]])
            pairs[key] = pairs.get(key, 0) + 1
    confusions = sorted(pairs.items(), key=lambda kv: -kv[1])[:40]

    save_json(REPORTS / f'05_test{suffix}.json', {'test_records': len(te), 'ensemble': ens['weights'], 'threshold': thr,
                                         'compared_with': args.compare,
                                         'test': results, 'test_by_origin': by_origin, 'val_oof_ensemble': val,
                                         'groups': groups,
                                         'confusions': [{'label': a, 'predicted': b, 'records': n} for (a, b), n in confusions]})
    xlsx = write_xlsx(te, Y, P, pred, classes, names, divn, ens, results, val, by_origin, groups, confusions, wrong,
                      len(tr), cols, label, REPORTS / f'05_test_report{suffix}.xlsx')

    say(f'Записів у test: {len(te)}. Поєднання: ' + ', '.join(f'{k} {v:g}' for k, v in ens['weights'].items() if v)
        + f'; поріг {thr}\n')
    keys = [('p_at_1', 'P@1'), ('p_at_1_other_ok', 'P@1 «Інше»'), ('hit_at_3', 'топ-3'), ('hit_at_5', 'топ-5'),
            ('micro_f1', 'micro-F1'), ('macro_f1', 'macro-F1'), ('division_p_at_1', 'розділ P@1')]
    say(f'{"test":34}' + ''.join(f'{h:>11}' for _, h in keys))
    for c in cols:
        say(f'{label(c)[:34]:34}' + ''.join(f'{results[c][k]:11.3f}' for k, _ in keys))
    say(f'{("val: " + label(args.ensemble))[:34]:34}' + ''.join(f'{val.get(k, np.nan):11.3f}' for k, _ in keys))
    say('P@1 «Інше»: як P@1, але для мітки «Інше …» (код на 99) годиться будь-яка група її розділу.')
    say('\nЗа походженням запису (поєднання):')
    for o, x in by_origin.items():
        say(f'  {o:13} ({x["records"]:5}): P@1 {x["p_at_1"]:.3f}, топ-3 {x["hit_at_3"]:.3f}, micro-F1 {x["micro_f1"]:.3f}')
    say('\nНайчастіші плутанини (мітка → перша група моделі):')
    for (a, b), n in confusions[:8]:
        say(f'  {n:4}  {a} {names.get(a, "")[:40]} → {b} {names.get(b, "")[:40]}')
    say(f'\nЗвіт: {xlsx} і {REPORTS / f"05_test{suffix}.json"}')


def write_xlsx(te, Y, P, pred, classes, names, divn, ens, results, val, by_origin, groups, confusions, wrong, n_train,
               cols, label, path):
    F = 'Arial'
    font, bold = Font(name=F, size=10), Font(name=F, size=10, bold=True)
    title, note = Font(name=F, size=13, bold=True), Font(name=F, size=9, italic=True, color='595959')
    head = PatternFill('solid', fgColor='D9E2F3')
    thin = Side(style='thin', color='BFBFBF')
    box = Border(top=thin, bottom=thin, left=thin, right=thin)
    wrap = Alignment(wrap_text=True, vertical='top')
    wb = Workbook()

    def table(ws, r0, heads, rows, fmts=None, widths=None):
        for k, h in enumerate(heads):
            c = ws.cell(r0, k + 1, h)
            c.font, c.fill, c.border, c.alignment = bold, head, box, wrap
        for i, row in enumerate(rows, 1):
            for k, v in enumerate(row):
                c = ws.cell(r0 + i, k + 1, v)
                c.font, c.border = font, box
                if fmts and fmts[k]:
                    c.number_format = fmts[k]
                if isinstance(v, str) and len(v) > 40:
                    c.alignment = wrap
        if widths:
            for k, w in enumerate(widths):
                ws.column_dimensions[get_column_letter(k + 1)].width = w
        return r0 + len(rows) + 1

    ws = wb.active
    ws.title = 'Підсумок'
    ws['A1'], ws['A1'].font = 'Класифікатор ANZSRC FoR 2020: остаточна оцінка на test', title
    ws['A2'] = (f'Test: {len(te)} записів із журналів і репозиторіїв, яких моделі не бачили (train — {n_train}). '
                'Ваги поєднання й пороги підібрано лише на val; на test нічого не налаштовувалось.')
    ws['A2'].font = note
    main = cols[len(ens['models'])]
    rows = [[text] + [results[c][key] for c in cols] + [val.get(key)] for key, text in METRICS]
    end = table(ws, 4, ['Метрика (test)'] + [label(c) for c in cols] + ['На val: ' + label(main)], rows,
                [None] + ['0.000'] * (len(cols) + 1), [44] + [15] * (len(cols) + 1))
    notes = ['Поєднання: ' + ', '.join(f'{label(k)} {v:g}' for k, v in ens['weights'].items() if v)
             + f'; спільний поріг {ens["threshold"]}; щонайменше одна група на запис.',
             'P@1 для «Інше …»: як P@1, але для мітки «Інше …» (код на 99) годиться будь-яка група її розділу.',
             'Мітки частково поставлено на весь журнал. За ручною перевіркою на val (04_error_review.xlsx) '
             'перша група підходить до тексту частіше, ніж збігається з міткою.']
    if any(c.startswith('було:') for c in cols):
        notes.insert(1, '«Було» — моделі, навчені на train до чистки міток; test для всіх той самий, мітки test не змінювались.')
    for k, text in enumerate(notes):
        ws.cell(end + 1 + k, 1, text).font = note
    r = end + len(notes) + 2
    ws.cell(r, 1, 'За походженням запису (поєднання)').font = bold
    table(ws, r + 1, ['Походження', 'Записів', 'P@1', 'Серед перших трьох', 'micro-F1', 'macro-F1', 'Перший розділ'],
          [[o, x['records'], x['p_at_1'], x['hit_at_3'], x['micro_f1'], x['macro_f1'], x['division_p_at_1']]
           for o, x in by_origin.items()], [None, '#,##0'] + ['0.000'] * 5)
    ws.sheet_view.showGridLines = False

    ws = wb.create_sheet('Групи')
    rows = [[g['code'], g['name'], f'{g["division"]} {divn.get(g["division"], "").capitalize()}', g['train'], g['test'],
             g['predicted'], g['precision'], g['recall'], g['f1'], g['ap']] for g in groups]
    rows.sort(key=lambda x: (x[8] is None, x[8] if x[8] is not None else 0))
    table(ws, 1, ['Код', 'Група', 'Розділ', 'У train', 'У test', 'Поставлено', 'Precision', 'Recall', 'F1', 'AP'], rows,
          [None, None, None, '#,##0', '#,##0', '#,##0', '0.00', '0.00', '0.00', '0.00'], [7, 52, 40, 9, 9, 11, 10, 9, 8, 8])
    ws.freeze_panes, ws.auto_filter.ref = 'A2', ws.dimensions
    ws.cell(len(rows) + 3, 1, 'Відсортовано від найгіршого F1. Групи з 1–10 записами в test оцінено ненадійно.').font = note

    ws = wb.create_sheet('Плутанини')
    rows = [[n, a, names.get(a, ''), b, names.get(b, ''), 'так' if a[:2] == b[:2] else 'ні'] for (a, b), n in confusions]
    table(ws, 1, ['Записів', 'Мітка', 'Назва мітки', 'Перша група моделі', 'Назва групи', 'Той самий розділ'], rows,
          ['#,##0', None, None, None, None, None], [9, 8, 46, 10, 46, 12])
    ws.freeze_panes = 'A2'

    ws = wb.create_sheet('Приклади помилок')
    rng = np.random.RandomState(0)
    pick = np.sort(rng.choice(wrong, min(100, len(wrong)), replace=False))
    order = np.argsort(-P, 1)[:, :3]
    rows = []
    for i in pick:
        x = te.iloc[i]
        rows.append([x.id, x.record_origin, x.source_title or x.source, x.title,
                     '; '.join(f'{c} {names.get(c, "")}' for c in x.labels.split(';')),
                     '; '.join(f'{classes[j]} {names.get(classes[j], "")} ({P[i, j]:.2f})' for j in order[i]),
                     x.abstract[:600]])
    table(ws, 1, ['Record_ID', 'Походження', 'Джерело', 'Назва', 'Мітки', 'Модель: перші три групи', 'Анотація (початок)'],
          rows, None, [22, 13, 28, 50, 44, 56, 80])
    ws.freeze_panes = 'A2'
    try:
        wb.save(path)
    except PermissionError:                       # the old report is open in Excel
        path = path.with_name(path.stem + '_new.xlsx')
        wb.save(path)
    return path


if __name__ == '__main__':
    main()
